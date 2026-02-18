from __future__ import annotations
from dataclasses import dataclass

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import global_mean_pool

@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 256
    dropout: int = 0.1    


class GCNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout,
    ):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        z = self.gcn(x, edge_index, edge_attr)
        z = F.relu(z)
        z = self.dropout(z)
        return x + z


class GCNModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig
    ):
        super().__init__()
        n = cfg.in_channels
        self.conv1 = GCNBlock(n, n, cfg.dropout)
        self.conv2 = GCNBlock(n, n, cfg.dropout)

    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x
