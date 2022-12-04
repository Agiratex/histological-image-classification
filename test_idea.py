import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils.train import train
from GaborLayer.GaborLayer import FourierGaborConv, GaborConv


transform = transforms.ToTensor()
device = 'cuda:0'
batch_size = 512

trainset = datasets.SVHN(root='./datasets', split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.SVHN(root='./datasets', split="test",
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

channels = 80
GCN = nn.Sequential(
    GaborConv(3, channels, 3, sigma = 1), #30
    nn.BatchNorm2d(channels),
    nn.ReLU(),
    GaborConv(channels, channels, 3, sigma = 1), #28
    nn.BatchNorm2d(channels),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), #14
    
    GaborConv(channels, 2*channels, 3, sigma = 1),#12
    nn.BatchNorm2d(2*channels),
    nn.ReLU(),
    GaborConv(2*channels, 2*channels, 3, sigma = 1), #10
    nn.BatchNorm2d(2*channels),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),#5
    
    GaborConv(2*channels, 4*channels, 3, sigma = 1),
    nn.BatchNorm2d(4*channels),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),

    nn.Flatten(1),
    nn.Dropout(0.2),
    nn.Linear(4*channels, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(GCN.parameters(), lr = 0.01)
writer = SummaryWriter(log_dir=f"svhn/Gabor3x3")
train(40, GCN, trainloader, testloader, optimizer, criterion, writer, device, "pretrained/1.pth")
print('Finished Training')