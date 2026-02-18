import os 
import os.path as osp
from typing import Optional, Callable, List


import torch 
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader 
import torch_geometric.transforms as T 
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



class SyntheticGraphCls(InMemoryDataset):
    """
    Practical custom InMemoryDataset:
      - raw/processed folders handled by PyG
      - process() creates a list[Data] then collates into big tensors
      - supports pre_transform (applied once at processing) and transform (applied on access)
    """
    def __init__(
        self,
        root: str,
        num_graphs: int = 2000,
        num_node_features: int = 16,
        num_classes: int = 4,
        min_nodes: int =20,
        max_nodes: int = 80,
        p_range=(0.05, 0.15),
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.num_graphs = num_graphs
        self._num_node_features = num_node_features
        self._num_classes = num_classes
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.p_range = p_range

        super().__init__(
            root, transform=transform, pre_transform=pre_transform,
            pre_filter=pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])


        @property
        def raw_file_names(self) -> List[str]:
            return ["_synthetic.marker"]
        
        @property
        def processed_file_names(self) -> List[str]:
            return ["data.pt"]
        
    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        marker = osp.join(self.raw_dir, self.raw_file_names[0])
        if not osp.exists(marker):
            with open(marker, "w") as f:
                f.write("ok")

    def process(self):
        data_list: List[Data] = []
        
        for _ in range(self.num_graphs):
            n = int(torch.randint(self.min_nodes, self.max_nodes+1, (1,)).item())
            p = float(torch.empty(1).uniform_(*self.p_range).item())
            data = make_random_graph(
                num_nodes=n,
                p=p,
                num_node_features=self._num_node_features,
                num_classes=self._num_classes
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    # Precompute things once at processing time:
    pre_transform = T.Compose([
        T.AddSelfLoops(),
        T.ToUndirected()
    ])


    # Applied each time you access an item (cheap, or randomized augmentations):
    transform = T.Compose([
        T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True,
                          add_negative_train_samples=False)
    ])


    dataset = SyntheticGraphCls(
        root = "synthetic_graph_cls",
        num_graphs=512, 
        num_node_features=32,
        num_classes=5, 
        pre_transform=pre_transform,
        transform=None
    )


    print(dataset)
    print("sample:", dataset[0])
    print("num features", dataset.num_node_features)

