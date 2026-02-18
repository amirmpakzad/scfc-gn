from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from scfc_gn.train.history import JsonlHistoryWriter


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    save_dir: str = "runs"

def train_loop(
        model: torch.nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        cfg: TrainConfig
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = JsonlHistoryWriter(str(save_dir/"history.jsonl"))
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        running_loss = 0.0
        n_batches = 0

        def check_tensor(name, t):
            if t is None:
                return
            ok = torch.isfinite(t).all().item()
            if not ok:
                bad = (~torch.isfinite(t)).sum().item()
                print(f"[BAD] {name}: non-finite count={bad}, min={t.min().item()}, max={t.max().item()}")
                raise RuntimeError(f"Non-finite in {name}")


        for batch in train_loader:
            batch = batch.to(device)

            check_tensor("batch.x", batch.x)
            check_tensor("batch.edge_attr", batch.edge_attr)
            check_tensor("batch.y", batch.y)

            opt.zero_grad()

            pred = model(
                x = batch.x,
                edge_index = batch.edge_index,
                edge_attr = batch.edge_attr,
            )
            loss = criterion(pred, batch.y)

            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            n_batches += 1

            if (global_step % cfg.log_interval) == 0:
                print(
                    f"epoch {epoch:03d} step {global_step:06d} | "
                    f"train loss {loss.item():.4f}"
                )
                history.log({
                    "type": "step",
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(loss.item()),
                })

            global_step += 1

        train_loss = running_loss / max(1, n_batches)
