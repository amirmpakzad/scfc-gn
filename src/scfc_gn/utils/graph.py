from __future__ import annotations 

from dataclasses import dataclass 
from typing import Optional, Tuple

import torch 

from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph 



def make_random_graph(
    num_nodes: int,
    p: float,
    num_node_features: int,
    num_classes: int
) -> Data:
    edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=p) #(2, E)
    x = torch.randn(num_nodes, num_node_features)
    y = torch.randint(0, num_classes, size = (1,), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data



@dataclass
class EdgeIndexResult:
    """small result object to avoid tuple soup."""
    edge_index: torch.Tensor 
    edge_attr: Optional[torch.Tensor] = None 


@dataclass(frozen=True)
class ConversionConfig:
    directed: bool = False
    include_self_loops: bool = False
    threshold: Optional[float] = None 
    keep_zero_weight_edges: bool = False 
    dtype_index: torch.dtype = torch.long 
    dtype_attr: torch.dtype = torch.float32 


class ConnectivityConverter:
    """Converts an adjacency / connectivity matrix into PyG-style edge_index."""
    def __init__(
        self,
        config: ConversionConfig = ConversionConfig()
    ) -> None:
        self._cfg = config 

    def to_edge_index(self, matrix : torch.Tensor) -> EdgeIndexResult:

        weights = self._extract_weights(matrix)
        row, col = self._select_edges(weights)

        edge_index = torch.stack([row, col], dim=0).to(self._cfg.dtype_index)
        edge_attr = weights[row, col].to(self._cfg.dtype_attr)

        return EdgeIndexResult(edge_index=edge_index, edge_attr=edge_attr)
    

    
    def _extract_weights(self, matrix: torch.Tensor) -> torch.Tensor:
        weights = matrix 

        if not self._cfg.directed:
            weights = (weights + weights.t()) / 2.0 

        if not self._cfg.include_self_loops:
            weights = weights.clone()
            weights.fill_diagonal_(0)

        return weights
    
    def _select_edges(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cfg.threshold is None:
            mask = weights != 0
        else:
            mask = weights.abs() > float(self._cfg.threshold)

        row, col = mask.nonzero(as_tuple=True)
        return row, col



def edge_index_from_connectivity(
    x: torch.ndarray,
    *,
    config: ConversionConfig = ConversionConfig(),
) -> Tuple[EdgeIndexResult, Optional[EdgeIndexResult]]:
    
    converter = ConnectivityConverter(config)
    x_res = converter.to_edge_index(x)
    return x_res



# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    x = torch.tensor([[0.0, 0.2, 0.0],
                  [0.2, 0.0, -0.5],
                  [0.0, -0.5, 0.0]], dtype=float)

    y = torch.tensor([[0.0, 0.0, 0.7],
                  [0.0, 0.0, 0.0],
                  [0.7, 0.0, 0.0]], dtype=float)

    cfg = ConversionConfig(directed=False, include_self_loops=False, threshold=1e-6)

    x_edges = edge_index_from_connectivity(x, config=cfg)
    y_edges = edge_index_from_connectivity(x, config=cfg)

    print("x edge_index:\n", x_edges.edge_index)
    print("x edge_attr:\n", x_edges.edge_attr)
    print("y edge_index:\n", y_edges.edge_index)
    print("y edge_attr:\n", y_edges.edge_attr)
