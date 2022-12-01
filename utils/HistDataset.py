import torch
from torch.utils.data import Dataset
from typing import Tuple
from torchvision import transforms
import numpy as np


#Dataset from (224, 224, 3) to (3, 224, 224) for torch convs
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
        # YOUR CODE HERE
        img = torch.Tensor(self.imgs[idx]).view(3, 224, 224)
        img = self.transform(img)
        return torch.FloatTensor(img), self.labels[idx]