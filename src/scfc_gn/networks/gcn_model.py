from __future__ import annotations
from dataclasses import dataclass

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

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
        in_channels = cfg.in_channels
        self.convs = nn.Sequential(
            GCNBlock(in_channels, int(in_channels/2), cfg.dropout),
            GCNBlock(int(in_channels/2), in_channels, cfg.dropout)
        )

    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.convs(x, edge_index, edge_attr)
        return x
