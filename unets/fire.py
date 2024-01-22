from collections import OrderedDict
import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes):
    	# 不改变输入特征图的宽高，只改变通道数
    	# 输入通道数为 inplanes，压缩通道数为 squeeze_planes，输出通道数为 2 * expand_planes
        super(Fire, self).__init__()
        # Squeeze 层
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Expand 层
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out