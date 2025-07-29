import random
import duckdb
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import PosixPath
from typing import Tuple

import modal
import torch
import torch_frame as tf
from torch_frame.data import Dataset
from torch_frame.utils import infer_df_stype
from torch_frame.data.loader import DataLoader
from torch_frame.nn.encoder import EmbeddingEncoder, LinearEncoder
from torch_frame.nn.models import FTTransformer
from torchmetrics import Accuracy, MeanAbsoluteError

# Modal app
app        = modal.App("parcel-hp-sweep")
MINUTES     = 60
HOURS       = 60 * MINUTES
GPU_TYPE    = "T4"            # change to H100 / A100 etc. if you like
vol         = modal.Volume.from_name("parcel")
VPATH       = PosixPath("/vol")
MODEL_DIR   = VPATH / "models"

# container images
base_image  = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "duckdb", "pyarrow", "pandas", "numpy"
    )
)
torch_image = (
    base_image
    .uv_pip_install(
        "torch",
        "torchvision",
        "pytorch-frame",
        "torchmetrics",
        "wandb",
        "tensorboard"
    )
)

# helpers
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class HParams:
    # model
    channels: int        = 32
    n_layers: int        = 10
    num_heads: int       = 8
    encoder_pad_size = 2
    attn_dropout = 0.3
    ffn_dropout = 0.3
    gamma: float = 0.95
    # optimisation
    lr: float            = 1e-2
    # dataset
    sample_rows: int     = 10
    batch_size: int      = 2
    val_split: float     = 0.2
    seed: int            = 42

# data module
def build_dataloaders(
    parquet_path: str,
    target_col: str,
    task: str,
    h: HParams,
) -> Tuple[DataLoader, DataLoader, tf.data.Dataset.col_stats, dict]:
    """
    Returns train & val DataLoaders plus col_stats / col_names_dict needed by model.
    `task` ∈ {"regression", "binary"}
    """
    q = f"""SELECT * FROM '{VPATH / parquet_path}'"""
    if h.sample_rows:
        q += f""" USING SAMPLE reservoir({h.sample_rows} ROWS) REPEATABLE ({h.seed})"""
    df = duckdb.sql(q).df()

    if task == "regression":
        df[target_col] = df[target_col].astype(float)
    else:
        df[target_col] = df[target_col].astype(int)

    # simple heuristic – drop obvious identifiers
    df = df.drop(columns=[c for c in df.columns if c.endswith("_id")], errors="ignore")

    col2stype = infer_df_stype(df)
    col2stype[target_col] = tf.numerical if task == "regression" else tf.categorical

    full_ds = Dataset(df, col_to_stype=col2stype, target_col=target_col)
    full_ds.materialize(path=VPATH / "full_stats.pt")   # now TensorFrame exists
    num_rows = full_ds.num_rows
    n_train  = int(num_rows * (1 - h.val_split))
    g        = torch.Generator().manual_seed(h.seed)
    perm     = torch.randperm(num_rows, generator=g)
    train_ds = full_ds.index_select(perm[:n_train])
    val_ds   = full_ds.index_select(perm[n_train:])

    train_ds.materialize(path=VPATH / "train_stats.pt")
    val_ds.materialize(path=VPATH / "val_stats.pt", col_stats=train_ds.col_stats)

    train_loader = DataLoader(train_ds, batch_size=h.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=h.batch_size)

    return train_loader, val_loader, train_ds.col_stats, train_ds.tensor_frame.col_names_dict


# training loop
def run_epoch(
    model, loader, optim, criterion, device
) -> float:
    model.train()
    total = n = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch).squeeze()
        loss  = criterion(pred, batch.y.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * len(batch.y)
        n += len(batch.y)
    return total / n

@torch.no_grad()
def evaluate(model, loader, metric, device) -> float:
    metric.reset()
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch).squeeze()
        if isinstance(metric, Accuracy):
            pred = torch.sigmoid(pred)
        metric.update(pred, batch.y.float())
    return metric.compute().item()


def build_model(model_name: str,
                h: HParams,
                col_stats,
                col_names_dict,
                device) -> torch.nn.Module:
    enc = {
        tf.stype.categorical: EmbeddingEncoder(),
        tf.stype.numerical:   LinearEncoder(),
    }

    if model_name == "ftt":
        return FTTransformer(
            channels=h.channels,
            num_layers=h.n_layers,
            out_channels=1,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=enc,
        ).to(device)

    elif model_name == "mlp":
        from torch_frame.nn.models import MLP
        return MLP(
            num_layers=h.n_layers,
            channels=h.channels,
            out_channels=1,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=enc,
        ).to(device)

    elif model_name == "tab_transformer":
        from torch_frame.nn.models import TabTransformer
        return TabTransformer(
            num_layers=h.n_layers,
            num_heads=h.num_heads,
            encoder_pad_size=h.encoder_pad_size,
            attn_dropout=h.attn_dropout,
            ffn_dropout=h.ffn_dropout,
            channels=h.channels,
            out_channels=1,
            col_stats=col_stats,
            col_names_dict=col_names_dict
        ).to(device)

    elif model_name == "tabnet":
        from torch_frame.nn.models import TabNet
        # TabNet expects **arrays**, so we wrap it with a tiny adapter
        return TabNet(
            out_channels=1,
            num_layers=h.n_layers,
            split_attn_channels=h.channels,
            split_feat_channels=h.channels,
            gamma=h.gamma,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
        ).to(device)

    else:
        raise ValueError(f"Unknown model '{model_name}'")


# remote fn
@app.function(
    gpu=GPU_TYPE,
    image=torch_image,
    timeout=2 * HOURS,
    volumes={VPATH: vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    retries=1
)
def train_model(
    parquet_path: str,
    target_col: str,
    task: str,
    h: HParams,
    run_to_end: bool,
    max_epochs: int = 50,
):
    """
    Train one model with given hyper-parameters.
    """
    set_seed(h.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data
    train_loader, val_loader, col_stats, col_names_dict = build_dataloaders(
        parquet_path, target_col, task, h
    )

    # ---- model
    enc = {
        tf.stype.categorical: EmbeddingEncoder(),
        tf.stype.numerical:   LinearEncoder(),
    }

    model = build_model("mlp", h, col_stats, col_names_dict, device=device)
    model = torch.compile(model, dynamic=True)

    # use data parellel for faster training
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # ---- optimisation
    optim = torch.optim.AdamW(model.parameters(), lr=h.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=h.gamma)

    criterion = torch.nn.SmoothL1Loss() if task == "regression" else torch.nn.BCEWithLogitsLoss()
    metric    = MeanAbsoluteError().to(device) if task == "regression" else Accuracy(task="binary").to(device)

    # W&B
    import wandb
    wandb.init(
        project="parcel-tabular",
        config=dict(**h.__dict__, task=task),
        name=f"E{datetime.now().strftime('%m%d_%H%M%S')}_ch{h.channels}_L{h.n_layers}_lr{h.lr:g}",
    )
    best_val, patience, PATIENCE = (np.inf if task=="regression" else 0.0), 0, 5

    for epoch in range(1, max_epochs + 1):
        tr_loss = run_epoch(model, train_loader, optim, criterion, device)

        # update LR after each epoch
        scheduler.step()
        # handy for W&B graphs
        current_lr = optim.param_groups[0]["lr"]

        val_metric = evaluate(model, val_loader, metric, device)
        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_metric": val_metric,
            "lr": current_lr
        })

        improved = (val_metric < best_val) if task=="regression" else (val_metric > best_val)
        if improved:
            best_val, patience = val_metric, 0
            # save weights into the shared Volume so other jobs can see them
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / f"{wandb.run.name}.pt")
            vol.commit()
        else:
            patience += 1
            if patience >= PATIENCE and not run_to_end:
                break
    wandb.finish()
    return float(best_val)

# sweep main
@app.local_entrypoint()
def main(
    parquet_path: str = "parcel_data.parquet",
    target_col: str   = "total_hours_from_receiving_to_last_success_delivery",
    task: str         = "regression",
):
    """
    Kick off an 8-run grid search, find the best, then resume it to completion.
    """

    # default hparams
    # ref = HParams()
    # ch_opts  = (16, ref.channels)
    # lyr_opts = (2,  ref.n_layers)
    # lr_opts  = (1e-3, ref.lr)

    # hp_list: List[HParams] = [
    #     HParams(channels=channel, n_layers=layer, lr=lr,
    #             sample_rows=ref.sample_rows,
    #             batch_size=ref.batch_size,
    #             val_split=ref.val_split,
    #             seed=42)
    #     for channel, layer, lr in product(ch_opts, lyr_opts, lr_opts)
    # ]

    # print(f"Launching {len(hp_list)} hyper-param jobs on Modal …")
    # # first round – early stop
    # early_results = train_model.starmap(
    #     [
    #         (parquet_path, target_col, task, h, False)   # run_to_end=False
    #         for h in hp_list
    #     ],
    #     order_outputs=False,
    # )

    # best_h, best_val = None, np.inf if task=="regression" else -np.inf
    # for res, h in zip(early_results, hp_list):
    #     v = float(res)
    #     better = (v < best_val) if task=="regression" else (v > best_val)
    #     print(f"{h} ⇒ {v:.4f}")
    #     if better:
    #         best_h, best_val = h, v

    hp = HParams()
    print(f"Launching {len([hp])} hyper-param jobs on Modal …")
    print(hp)
    best_val = train_model.remote(parquet_path, target_col, task, hp, True, max_epochs=200)
    best_h = hp

    print(f"\nBest so far: {best_h} ({best_val:.4f}) — continuing to full training …")
    # train_model.remote(parquet_path, target_col, task, best_h, True, max_epochs=100)