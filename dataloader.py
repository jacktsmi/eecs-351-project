import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.tensor(data)
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
    
    def __len__(self):
        return min(len(d) for d in self.datasets)