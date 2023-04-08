import torch
from torch.fft import fft2, ifft2
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from cv2 import getGaborKernel
from typing import Tuple
import numpy as np


def getGaborFilterBank(size : Tuple[int,int] = (3,3), sigma = 1.0, n_filters = 4, lam = 2, gamma = 0.1, shape = [0], device = 'cpu') -> torch.Tensor:
    weight = torch.zeros([n_filters, *shape]).to(device)

    sz_x = shape[2]

    sz_y = shape[3]
    
    for orientation in range(n_filters):
        theta = np.pi / n_filters * orientation

        sigma_x = sigma
        sigma_y = sigma / gamma

        # Bounding box
        nstds = 3  # Number of standard deviation sigma
        y, x = np.meshgrid(torch.arange(-np.ceil(sz_y/2) + 1, np.ceil(sz_y/2)), torch.arange(-np.ceil(sz_x/2) + 1, np.ceil(sz_x/2)))

        x = torch.cat([torch.cat([torch.Tensor(x).to(device).unsqueeze(0)]*shape[1]).unsqueeze(0)]*shape[0])
        y = torch.cat([torch.cat([torch.Tensor(y).to(device).unsqueeze(0)]*shape[1]).unsqueeze(0)]*shape[0])

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        ex = torch.exp(
            -0.5 * (x_theta.to(device)**2/sigma_x.to(device).view(*sigma_x.shape, 1, 1)**2 + y_theta.to(device)**2 / sigma_y.to(device).view(*sigma_y.shape, 1, 1)**2)
        ).to(device)


        cos = torch.cos(2 * np.pi / lam.view(*sigma_y.shape, 1, 1) * x_theta.to(device)).to(device)

        weight[orientation] = ex * cos

    return weight.to(device)


class GaborConv(_ConvNd):
    """My Gabor Convolution Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, padding_mode="zeros", dilation=1, groups=1, 
                    bias=False,
                    sigma : float = 1.0, n_filters : int = 4, 
                    lam : float = 2, gamma : float = 0.1):

        if groups != 1:
            raise ValueError('Group-conv not supported!')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GaborConv, self).__init__(
            in_channels, out_channels//n_filters, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        self.sigma = torch.nn.parameter.Parameter(data = torch.ones(self.weight.shape[:2]))
        self.lam = torch.nn.parameter.Parameter(data = torch.ones(self.weight.shape[:2]))
        self.gamma = torch.nn.parameter.Parameter(data = torch.ones(self.weight.shape[:2]))
        

    def forward(self, x):
        
        self.gabor_bank = getGaborFilterBank(self.kernel_size, self.sigma, self.n_filters, 
                                                self.lam, self.gamma, self.weight.shape, self.weight.device)
        pad = self.kernel_size[0]//2

        new_weight = F.pad(self.weight, (pad,pad,pad,pad), mode='constant').to(self.weight.device)

        new_weight = new_weight.view(1, -1, new_weight.shape[-2], new_weight.shape[-1]).to(self.weight.device)

        new_weight = F.conv2d(new_weight, self.gabor_bank.to(self.weight.device).view(-1, 1, *self.kernel_size), groups=new_weight.shape[1]).to(self.weight.device)

        new_weight = new_weight.view(self.out_channels*self.n_filters, self.in_channels, *self.kernel_size).to(self.weight.device)

        y = F.conv2d(x, new_weight, padding = self.padding, stride = self.stride).to(self.weight.device)
        
        return y


class FourierGaborConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, padding_mode="zeros", dilation=1, groups=1, 
                    bias=False,
                    sigma : float = 1.0, n_filters : int = 4, 
                    lam : float = 2, gamma : float = 0.1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(FourierGaborConv, self).__init__(
            in_channels, out_channels//n_filters, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        pad = (16 - kernel_size[0]) // 2
        self.pad = pad
        self.g_f_bank = fft2(F.pad(getGaborFilterBank(kernel_size, sigma, n_filters, 
                                                lam, gamma), (0, pad, 0, pad)))

            
    def forward(self, x): #16,1,28,28
        weight_f = fft2(F.pad(self.weight, (0, self.pad, 0, self.pad)))
        new_weight = ifft2(torch.mul(weight_f.view(1, -1, self.kernel_size[0] + self.pad, 
                        self.kernel_size[1] + self.pad), self.g_f_bank.to(x.device))).real
        new_weight = new_weight[..., self.pad:, self.pad:]
        new_weight = new_weight.view(-1, 
                    self.weight.size()[1], self.weight.size()[2], self.weight.size()[3])
        y = F.conv2d(x, new_weight, stride=self.stride, padding=self.padding)
        return y