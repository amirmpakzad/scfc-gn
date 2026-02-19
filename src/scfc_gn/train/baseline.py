import torch
import torch.nn.functional as F


def compute_mean_fc(train_loader, device):
    total = 0
    count = 0

    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            total += batch.y.sum(dim=0)
            count += batch.num_graphs

    return total / count



def mean_fc_baseline_loss(loader, mean_fc, device):
    total = 0.0
    n_elems = 0

    with torch.no_grad():
        for batch in loader:
            y = batch.y.to(device)  # [B, ...]
            pred = mean_fc.expand_as(y)  # repeat mean template for the batch

            total += F.mse_loss(pred, y, reduction="sum").item()
            n_elems += y.numel()

    return total / n_elems
