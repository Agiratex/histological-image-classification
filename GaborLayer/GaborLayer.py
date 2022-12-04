import torch
from torch.fft import fft2, ifft2
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from cv2 import getGaborKernel
from typing import Tuple
import numpy as np


def getGaborFilterBank(size : Tuple[int,int] = (3,3), sigma : float = 1.0, n_filters : int = 4, lam : float = 2, gamma : float = 0.1) -> torch.Tensor:
    bank = []
    for i in range(n_filters):
        bank.append(getGaborKernel(size, sigma, i*np.pi/n_filters, lam, gamma, 0))
    return torch.Tensor(np.array(bank)).unsqueeze(1)


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
        self.gabor_bank = getGaborFilterBank(kernel_size, sigma, n_filters, 
                                                lam, gamma)

    def forward(self, x):
        pad = self.kernel_size[0]//2
        new_weight = F.pad(self.weight.view(-1, 1, self.weight.size()[-1], 
                            self.weight.size()[-1]), (pad,pad,pad,pad), mode='constant')
        new_weight = F.conv2d(new_weight, self.gabor_bank.to(new_weight.device)).view(-1, 
                    self.weight.size()[1], self.weight.size()[2], self.weight.size()[3])
        new_weight
        y = F.conv2d(x, new_weight, padding = self.padding, stride = self.stride)
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