# src/utils/nav_dataset.py
import torch as t
from torch.utils.data import Dataset

class NavigationTrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = t.FloatTensor(x)
        y = t.FloatTensor(y)
        return x, y

