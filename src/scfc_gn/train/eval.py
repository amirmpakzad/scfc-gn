from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
@dataclass
class EvalResult:
    loss: float

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str
):
    model.to(device)
    model.eval()
    total_loss = 0.0
    n = 0
    criterion = nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index

        pred = model(x, edge_index)
        loss = criterion(pred, y)

        total_loss += float(loss.item())
        n += 1

    if n == 0:
        return EvalResult(loss=0)

    return EvalResult(loss=total_loss / n)
