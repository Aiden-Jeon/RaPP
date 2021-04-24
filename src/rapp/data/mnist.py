from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import MNIST
from torchvision import transforms as T
import pytorch_lightning as pl

from .dataset import CustomDataset, _flatten, _normalize


class MNISTDataModule(pl.LightningDataModule):
    name = "mnist"

    def __init__(
        self,
        data_dir: str = "",
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 256,
        unseen_label: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.unseen_label = unseen_label
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
        self.test_transforms = self.default_transforms

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Saves MNIST files to `data_dir`"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""
        # get all dataset
        extra = dict(transform=self.default_transforms)
        train_dataset = MNIST(self.data_dir, train=True, download=False)
        test_dataset = MNIST(self.data_dir, train=False, download=False)
        train_data, train_label = train_dataset.data, train_dataset.targets
        test_data, test_label = test_dataset.data, test_dataset.targets
        data = torch.cat([train_data, test_data])
        labels = torch.cat([train_label, test_label])

        # split data with seen labels and unseen labels
        seen_idx = labels != self.unseen_label
        unseen_idx = labels == self.unseen_label
        seen_data = data[seen_idx]
        unseen_data = data[unseen_idx]
        seen_dataset = CustomDataset(
            seen_data, torch.Tensor([0] * len(seen_data)), **extra
        )
        unseen_dataset = CustomDataset(
            unseen_data, torch.Tensor([1] * len(unseen_data)), **extra
        )

        # split seen data to train, valid, test
        train_size = int(seen_data.size(0) * 0.7)
        valid_size = int(seen_data.size(0) * 0.2)
        test_size = len(seen_data) - train_size - valid_size

        self.dataset_train, self.dataset_val, test_data = random_split(
            seen_dataset, [train_size, valid_size, test_size]
        )
        # make test data with seen data and unseen data
        self.dataset_test = ConcatDataset([test_data, unseen_dataset])

    def train_dataloader(self):
        """MNIST train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """MNIST val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        """MNIST test set uses the test split"""
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    @property
    def default_transforms(self):
        transforms = []
        if self.normalize:
            transforms.append(T.Lambda(_normalize))
        transforms.append(T.Lambda(_flatten))
        transforms = T.Compose(transforms)
        return transforms
