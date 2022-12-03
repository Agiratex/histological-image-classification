import torch
from torch.utils.data import Dataset
from typing import Tuple
from torchvision import transforms
import numpy as np
import gdown

def download_dataset( path : str, test: bool = False):
    if test:
        gdown.download(f'https://drive.google.com/uc?id=1RfPou3pFKpuHDJZ-D9XDFzgvwpUBFlDr', path, quiet=False)
    else:
        gdown.download(f'https://drive.google.com/uc?id=1XtQzVQ5XbrfxpLHJuL0XBGJ5U7CS-cLi', path, quiet=False)


class HistDataset(Dataset):

    def __init__(
        self,
        path : str,
        transform : transforms,
    ):
        data = np.load(path)
        self.imgs = data['data']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.FloatTensor, int]:
        img = self.transform(self.imgs[idx])
        return torch.FloatTensor(img), self.labels[idx]