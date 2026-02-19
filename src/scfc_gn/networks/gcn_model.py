from __future__ import annotations
from dataclasses import dataclass

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 256
    dropout: float = 0.1


class GCNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout,
    ):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        z = self.gcn(x, edge_index)
        z = F.relu(z)
        z = self.dropout(z)
        return z


class GCNModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig
    ):
        super().__init__()
        n = cfg.in_channels
        self.conv1 = GCNBlock(n, n, cfg.dropout)
        self.conv2 = GCNBlock(n, n, cfg.dropout)
        self.decoder = Decoder(d=n)

    
    def forward(self, x, edge_index, edge_attr=None):
        z = self.conv1(x, edge_index, edge_attr)
        z = self.conv2(z, edge_index, edge_attr)
        y = self.decoder(z, edge_index)
        return y


class Decoder(nn.Module):
    def __init__(
            self,
            d,
            hidden = 256,
    ):
        super().__init__()
        self.w = nn.Parameter(torch.rand(d, d) * 0.01)
        self.bias = nn.Parameter(torch.zeros((d, d)))

    def forward(self, z, edge_index):
        y_hat = z @ self.w @ z.t()
        y_hat = y_hat + self.bias
        return y_hat

