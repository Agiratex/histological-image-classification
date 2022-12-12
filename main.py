import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils.HistDataset import HistDataset, download_dataset
from utils.train import train
from GaborLayer.GaborLayer import FourierGaborConv, GaborConv

# download_dataset("datasets/train.npz", test=False)
# download_dataset("datasets/test.npz", test=True)

class ConcGCN(nn.Module):
    def __init__(self):
        super(ConcGCN, self).__init__()
        channels = 44
        self.GCN11 = nn.Sequential(
            FourierGaborConv(3, channels, 11, sigma = 5, n_filters=4), #214
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            FourierGaborConv(channels, channels, 11, sigma = 5, n_filters=4), #204
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #102
            
            FourierGaborConv(channels, 2*channels, 11, sigma = 5, n_filters=4),#92
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            FourierGaborConv(2*channels, 2*channels, 11, sigma = 5, n_filters=4), #82
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#41

            FourierGaborConv(2*channels, 4*channels, 11, sigma = 5, n_filters=4), #31
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            FourierGaborConv(4*channels, 4*channels, 11, sigma = 5, padding = 1, n_filters=4), #23
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 11

            FourierGaborConv(4*channels, 8*channels, 11, sigma = 5, n_filters=4),#
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),

            nn.Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(8*channels, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        channels = 48
        self.GCN7 = nn.Sequential(
            GaborConv(3, channels, 7, sigma = 2, n_filters=4), #218
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            GaborConv(channels, channels, 7, sigma = 2, n_filters=4), #212
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #106
            
            GaborConv(channels, 2*channels, 7, sigma = 2, n_filters=4),#100
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            GaborConv(2*channels, 2*channels, 7, sigma = 2, n_filters=4), #94
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#47

            GaborConv(2*channels, 4*channels, 7, sigma = 2, n_filters=4), #41
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            GaborConv(4*channels, 4*channels, 7, sigma = 2, n_filters=4), #35
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 17
            
            GaborConv(4*channels, 8*channels, 7, sigma = 2, n_filters=4), #11
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            GaborConv(8*channels, 8*channels, 7, sigma = 2, n_filters=4), #5
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #2

            nn.Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(8*4*channels, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        channels = 36
        self.GCN9 = nn.Sequential(
            GaborConv(3, channels, 9, padding = 1, sigma = 2, n_filters=4), #218
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            GaborConv(channels, channels, 9, padding = 1, sigma = 2, n_filters=4), #212
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #106
            
            GaborConv(channels, 2*channels, 9, padding = 1, sigma = 2, n_filters=4),#100
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            GaborConv(2*channels, 2*channels, 9, padding = 1, sigma = 2, n_filters=4), #94
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#47

            GaborConv(2*channels, 4*channels, 9, padding = 1, sigma = 2, n_filters=4), #41
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            GaborConv(4*channels, 4*channels, 9, padding = 1, sigma = 2, n_filters=4), #35
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 17
            
            GaborConv(4*channels, 8*channels, 9, padding = 1, sigma = 2, n_filters=4), #11
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            GaborConv(8*channels, 8*channels, 9, padding = 1, sigma = 2, n_filters=4), #5
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #2

            nn.Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(8*4*channels, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        channels = 48
        self.GCN3 = nn.Sequential(
            GaborConv(3, channels, 3, sigma = 1, n_filters=4), #222
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            GaborConv(channels, channels, 3, sigma = 1, n_filters=4), #220
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #110
            
            GaborConv(channels, 2*channels, 3, sigma = 1, n_filters=4),#108
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            GaborConv(2*channels, 2*channels, 3, sigma = 1, n_filters=4), #106
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#53

            GaborConv(2*channels, 4*channels, 3, sigma = 1, n_filters=4), #51
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            GaborConv(4*channels, 4*channels, 3, sigma = 1, n_filters=4), #49
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 24
            
            GaborConv(4*channels, 8*channels, 3, sigma = 1, n_filters=4), #22
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            GaborConv(8*channels, 8*channels, 3, sigma = 1, n_filters=4), #20
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 10

            GaborConv(8*channels, 16*channels, 3, sigma = 2, n_filters=4),# 8
            nn.BatchNorm2d(16*channels),
            nn.ReLU(),
            GaborConv(16*channels, 16*channels, 3, sigma = 2, n_filters=4),# 6
            nn.BatchNorm2d(16*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 3

            nn.Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(16*9*channels, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        channels = 32
        self.GCN5 = nn.Sequential(
            GaborConv(3, channels, 5, sigma = 2, padding = 1, n_filters=4), #222
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            GaborConv(channels, channels, 5, sigma = 2, padding = 1, n_filters=4), #220
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #110
            
            GaborConv(channels, 2*channels, 5, sigma = 2, padding = 1, n_filters=4),#108
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            GaborConv(2*channels, 2*channels, 5, sigma = 2, padding = 1, n_filters=4), #106
            nn.BatchNorm2d(2*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#53

            GaborConv(2*channels, 4*channels, 5, sigma = 2, padding = 1, n_filters=4), #51
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            GaborConv(4*channels, 4*channels, 5, sigma = 2, padding = 1, n_filters=4), #49
            nn.BatchNorm2d(4*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 24
            
            GaborConv(4*channels, 8*channels, 5, sigma = 2, padding = 1, n_filters=4), #22
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            GaborConv(8*channels, 8*channels, 5, sigma = 2, padding = 1, n_filters=4), #20
            nn.BatchNorm2d(8*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 10

            GaborConv(8*channels, 16*channels, 5, sigma = 2, padding = 1, n_filters=4),# 8
            nn.BatchNorm2d(16*channels),
            nn.ReLU(),
            GaborConv(16*channels, 16*channels, 5, sigma = 2, padding = 1, n_filters=4),# 6
            nn.BatchNorm2d(16*channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #3

            nn.Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(16*9*channels, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        self.GCN11.load_state_dict(torch.load("pretrained/HistGab11x11/96.pth"))
        self.GCN9.load_state_dict(torch.load("pretrained/HistGab9x9/95.pth"))
        self.GCN7.load_state_dict(torch.load("pretrained/HistGab7x7/68.pth"))
        self.GCN5.load_state_dict(torch.load("pretrained/HistGab5x5/86.pth"))
        self.GCN3.load_state_dict(torch.load("pretrained/HistGab3x3/87.pth"))
        self.w = torch.nn.parameter.Parameter(torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
        
    def forward(self, x):
        with torch.no_grad():
            x11 = torch.sigmoid(self.GCN11(x))
            x9 = torch.sigmoid(self.GCN9(x))
            x7 = torch.sigmoid(self.GCN7(x))
            x5 = torch.sigmoid(self.GCN5(x))
            x3 = torch.sigmoid(self.GCN3(x))
        x = x3 * self.w[0] + x5 * self.w[1] + x7 * self.w[2] + x9 * self.w[3] + \
            x11 * self.w[4]
        
        return x

device = "cuda:1"
batch_size = 128

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])

# train_dataset = HistDataset('datasets/train.npz', transform)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
# test_dataset = HistDataset('datasets/test.npz', transform)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                           shuffle=False, num_workers=2)

device = 'cuda:1'
GCN = ConcGCN().to(device)

GCN.load_state_dict(torch.load("/home/e.murin/histological-image-classification/pretrained/ConcGCN5.pth"))

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(GCN.parameters(), lr = 0.01)
# writer = SummaryWriter(log_dir=f"logs/Hist/ConcGCN")
# train(10, GCN, trainloader, testloader, optimizer, criterion, writer, device, "pretrained/ConcGCN")
# print('Finished Training')

print(GCN.w)