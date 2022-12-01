import torch
from torch import nn
from torchvision import transforms
from torchsummary import summary
from utils.HistDataset import HistDataset
from GaborLayer.GaborLayer import GaborConv

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64

# train_dataset = HistDataset('datasets/train.npz', transform)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
# test_dataset = HistDataset('datasets/test.npz', transform)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                           shuffle=False, num_workers=2)

channels = 24
GCN = nn.Sequential(
    GaborConv(3, channels, 5, sigma = 2, padding = 1), #222
    nn.BatchNorm2d(channels),
    nn.ReLU(),
    GaborConv(channels, channels, 5, sigma = 2, padding = 1), #220
    nn.BatchNorm2d(channels),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), #110
    
    GaborConv(channels, 2*channels, 5, sigma = 2, padding = 1),#108
    nn.BatchNorm2d(2*channels),
    nn.ReLU(),
    GaborConv(2*channels, 2*channels, 5, sigma = 2, padding = 1), #106
    nn.BatchNorm2d(2*channels),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),#53

    GaborConv(2*channels, 4*channels, 5, sigma = 2, padding = 1),#51
    nn.BatchNorm2d(4*channels),
    nn.ReLU(),
    GaborConv(4*channels, 4*channels, 5, sigma = 2, padding = 1), #49
    nn.BatchNorm2d(4*channels),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # 24

    GaborConv(4*channels, 8*channels, 5, sigma = 2, padding = 1),#22
    nn.BatchNorm2d(8*channels),
    nn.ReLU(),
    GaborConv(8*channels, 8*channels, 5, sigma = 2, padding = 1), #20
    nn.BatchNorm2d(8*channels),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), #10

    GaborConv(8*channels, 16*channels, 5, sigma = 2, padding = 1),#8
    nn.BatchNorm2d(16*channels),
    nn.ReLU(),
    GaborConv(16*channels, 16*channels, 5, sigma = 2, padding = 1),#6
    nn.BatchNorm2d(16*channels),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), #3

    GaborConv(16*channels, 32*channels, 5, sigma = 2, padding = 1),#1
    nn.BatchNorm2d(32*channels),
    nn.ReLU(),

    nn.Flatten(1),
    nn.Linear(32*channels, 128),
    nn.ReLU(),
    nn.Linear(128, 9)
).to(device)

summary(GCN, (3, 224, 224))