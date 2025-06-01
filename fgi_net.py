import math, os, random, time, warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class ConvBN_Swish(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)  # Swish

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)  # Swish
        return x


class FGINet(nn.Module):
    def __init__(self):
        super.__init__()
