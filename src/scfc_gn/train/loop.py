from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from scfc_gn.train.eval import evaluate
from scfc_gn.train.history import JsonlHistoryWriter
from scfc_gn.train.baseline import compute_mean_fc, mean_fc_baseline_loss


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    save_dir: str = "runs"


@dataclass
class TrainResult:
    dir : str
    last_pred_mean: float = None
    last_pred_std: float = None
    y_mean: float = None
    y_std: float = None
    pearson_corr: float = None
    baseline: float = None
    ckpt_path: str = None


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: str,
    cfg: TrainConfig,
    val_loader: Optional[DataLoader] = None,
):
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = JsonlHistoryWriter(str(save_dir / "history.jsonl"))
    global_step = 0
    best_val = float("inf")
    last_pred = None
    last_batch = None
    train_result = TrainResult(
        dir = str(save_dir / "history.jsonl")
    )


    for epoch in range(1, cfg.epochs + 1):
        model.train()

        loss_sum = 0.0
        n_graphs = 0

        for batch in train_loader:
            batch = batch.to(device)
            bs = batch.num_graphs

            opt.zero_grad()

            pred = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
            )

            loss = criterion(pred, batch.y)
            loss.backward()


            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()
            loss_sum += loss.item() * bs
            n_graphs += bs

            if (global_step % cfg.log_interval) == 0:
                history.log({
                    "type": "step",
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(loss.item()),
                })

            global_step += 1


            if epoch == cfg.epochs:
                last_pred = pred
                last_batch = batch

        train_loss = loss_sum / max(1, n_graphs)

        # validation
        if val_loader is not None:
            val = evaluate(model, val_loader, device)
            val_loss = float(val.loss)
            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )
        else:
            val_loss = float("nan")
            print(f"epoch {epoch:03d} | train loss {train_loss:.4f}")

        history.log({
            "type": "epoch",
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        mean_fc= compute_mean_fc(train_loader, device)
        train_mean_baseline = mean_fc_baseline_loss(train_loader, mean_fc, device)


        if epoch == cfg.epochs and last_pred is not None and last_batch is not None:
            with torch.no_grad():
                p = last_pred.detach().cpu().reshape(-1).numpy()
                y = last_batch.y.detach().cpu().reshape(-1).numpy()

                # handle degenerate case (std=0) to avoid NaN correlation
                if p.std() == 0.0 or y.std() == 0.0:
                    corr = float("nan")
                else:
                    corr = float(np.corrcoef(p, y)[0, 1])

                train_result.last_pred_mean = float(p.mean())
                train_result.last_pred_std = float(p.std())
                train_result.y_mean = float(y.mean())
                train_result.y_std = float(y.std())
                train_result.pearson_corr = corr
                train_result.baseline_loss = train_mean_baseline
            # checkpoints
        ckpt_path = save_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            best_path = save_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": best_val}, best_path)
            print(f"  saved best -> {best_path}")

        train_result.ckpt_path = best_path
    return train_result