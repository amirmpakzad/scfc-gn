from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np


@dataclass
class TestResult:
    loss: float
    pearson_corr: float
    last_y: np.ndarray
    last_pred: np.ndarray


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    return epoch

def test_loop(
        model: nn.Module,
        loader: DataLoader,
        device: str,
):
    model.to(device)
    model.eval()

    total_loss = 0.0
    n = 0
    last_y = None
    last_pred = None

    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index

        pred = model(x, edge_index)
        loss = F.mse_loss(pred, y)

        total_loss += float(loss.item())
        n+=1
        last_y = y
        last_pred = pred

    last_y = last_y.cpu().numpy()
    last_pred = last_pred.cpu().numpy()
    corr = np.corrcoef(last_y, last_pred)[0, 1]

    return TestResult(
        loss = total_loss / len(loader),
        pearson_corr = corr,
        last_pred = last_pred,
        last_y = last_y
    )


