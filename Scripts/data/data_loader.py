import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RailwayDataset(Dataset):
    def __init__(self, insar_data, labels, transform=None, augment=None):
        self.insar_data = insar_data
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.insar_data[idx]
        label = self.labels[idx]
        if self.augment:
            sample = self.augment(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def get_dataloader(insar_data, labels, batch_size, shuffle=True, augment=None):
    dataset = RailwayDataset(insar_data, labels, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def split_data(data, labels, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    return data[train_idx], labels[train_idx], data[val_idx], labels[val_idx]
