from typing import List

import torch 
from torch_geometric.data import Data 
from torch_geometric.loader import DataLoader 



def convert(
        x: torch.Tensor,
        y: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
        ):
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

    
    
def get_loader(graphs, batch_size, shuffle:int = False):
    data = Data(
        x=item.x, 
        edge_index=item.edge_index, 
        edge_attr=item.edge_attr, 
        y=item.y for item in graphs)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader