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
"""

import torch
from torch import nn
from torch.nn import functional as F

from utils.GCN import GatedSpatialConv2d
from utils.utils_model import res,Conv_block,BasicBlock


class Punet(nn.Module):
    def __init__(self):
        super(Punet, self).__init__()

        self.down_conv0  = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.down_conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(32, 64),
        )
        self.down_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(64, 128),
        )
        self.down_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(128, 256),
        )
        self.down_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(256, 512),
        )
        self.down_conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(512, 1024),
        )

        self.out_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1)
        self.res_1=res(64)
        self.res_2=res(128)
        self.res_3=res(256)
        self.res_4=res(512)

        
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1_later = Conv_block(1024, 512)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2_later = Conv_block(512, 256)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3_later = Conv_block(256, 128)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4_later = Conv_block(128, 64)


        self.up_conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_conv5_later =  nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2)
        )

        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)


        self.res1 = BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = BasicBlock(16, 16, stride=1, downsample=None)
        
        

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(32, 32)
        self.gate2 = GatedSpatialConv2d(16, 16)
        self.out_conv_=nn.Sequential(nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=2,stride=2),
            nn.ReLU())

        self.conv_bound=nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1)
        
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):

        features_stack = []
        

        x_lose=self.down_conv0(x)

        x = self.down_conv1(x_lose)


        
        x_size=x.size()

        x_1=x.clone()

        features_stack.append(self.res_1(x))
        
        x = self.down_conv2(x)
        features_stack.append(self.res_2(x))
        x = self.down_conv3(x)
        x_3=x.clone()
        features_stack.append(self.res_3(x))
        x = self.down_conv4(x)
        x_4=x.clone()
        features_stack.append(self.res_4(x))
        
        x = self.down_conv5(x)

        
        xb = F.interpolate(self.dsn3(x_3), x_size[2:],
                            mode='bilinear', align_corners=True)
        xc = F.interpolate(self.dsn4(x_4), x_size[2:],
                            mode='bilinear', align_corners=True)
        
        m1f = F.interpolate(x_1, x_size[2:], mode='bilinear', align_corners=True)


        cs = self.res1(m1f)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, xb)
        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, xc)
        cs = self.res3(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)

        out_border = cs

        out_border=self.out_conv_(out_border)

        out_border_out= self.conv_bound(out_border)

        out_border_out= self.sigmoid(out_border_out)



        x = self.up_conv1(x)
        
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv1_later(x)

        x = self.up_conv2(x)
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv2_later(x)

        x = self.up_conv3(x)
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv3_later(x)

        x = self.up_conv4(x)
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv4_later(x)

        x = self.up_conv5(x)
        
        x = self.up_conv5_later(x)



        out1 = self.out_conv(x)


        

        x = torch.cat((x_lose, out1,out_border), dim=1)
        
        return x,out_border_out
