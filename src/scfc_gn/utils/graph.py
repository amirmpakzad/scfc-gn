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

