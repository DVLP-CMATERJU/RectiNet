"""
Official Code Implementation of:
"A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping"
Authors:    Hmrishav Bandyopadhyay,
            Tanmoy Dasgupta,
            Nibaran Das,
            Mita Nasipuri

Code: Hmrishav Bandyopadhyay

Code references:
>>>https://github.com/wuleiaty/DocUNet
>>>https://github.com/nv-tlabs/GSCNN
"""

from torch import nn
import torch.nn.functional as F

def initialize_weights(*models):
   for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight,a=0,mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def conv3x3(in_planes, out_planes, stride=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class res(nn.Module):
    def __init__(self,in_channels):
        super(res,self).__init__()
        self.c1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True))
        
    def forward(self,x):
        x1=self.c1(x)
        return x1
