from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

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
        running_acc = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()

            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(pred, batch.y)

            loss.backward()
            opt.step()

            #acc = accuracy(pred.detach(), y)

        print("here it arrived")