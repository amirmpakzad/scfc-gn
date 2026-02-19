import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader


class SCFCDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]