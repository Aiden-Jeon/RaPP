import torch
from torch.utils.data import Dataset


def _flatten(x):
    return x.flatten()


def _normalize(x):
    return x / 255


class CustomDataset(Dataset):
    def __init__(self, data: torch.Tensor, label: torch.Tensor, transform: callable):
        super().__init__()
        assert data.size(0) == label.size(0), "Size mismatch between tensors"
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return self.label.size(0)
