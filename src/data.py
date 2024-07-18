import pandas as pd
import gzip
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader

# Read the FashionMNIST images
def read_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Read the magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # Read the number of images
        num_images = int.from_bytes(f.read(4), 'big')
        # Read the number of rows and columns
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)

    return torch.tensor(images, dtype=torch.float32)

# Read the FashionMNIST labels
def read_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Read the magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # Read the number of items
        num_items = int.from_bytes(f.read(4), 'big')
        
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return torch.tensor(labels, dtype=torch.long)

# Generic Dataset class
class GenericDataset(Dataset):
    def __init__(self, ft_path, tg_path):
        self.x, self.y = self.load(ft_path, tg_path)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)

    def load(ft_path, tg_path):
        ft = pd.read_csv(ft_path, header = None, sep = " " )
        tg = pd.read_csv(tg_path, header = None, sep = " " )
        return ft, tg

# Fashion Dataset class
class FashionDataset(Dataset):
    def __init__(self, ft_path, tg_path):
        self.ft_path = ft_path
        self.tg_path = tg_path
        self.x, self.y = self.load()

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)

    def load(self):
        images = read_mnist_images(self.ft_path)
        targets = read_mnist_labels(self.tg_path)
        return images, targets
    
    def get_in_out_size(self):
        return self.x.shape[1], len(self.y.unique())

# Fashion DataLoader class
class FashionDataLoader:
    def __init__(
        self, 
        dataset: FashionDataset, 
        batch_size: int,
        shuffle: bool
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        # Estimate num batches in loader
        return len(self.dataset) // self.batch_size

    def load(self) -> torch.Tensor:
        dl = DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=self.shuffle
                )
        return dl
