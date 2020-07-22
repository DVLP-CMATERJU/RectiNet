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

from utils.utils_model import res,Conv_block,BasicBlock



class Sunet(nn.Module):
    def __init__(self):
        super(Sunet, self).__init__()

        self.down2_conv0=Conv_block(50, 32)

        self.down2_conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(32, 64),
        )

        self.down2_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(64, 128),
        )
        self.down2_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(128, 256),
        )
        self.down2_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(256, 512),
        )
        self.down2_conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(512, 1024),
        )

        self.down2_conv6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(1024, 2048),
        )

        self.res_1_=res(64)
        self.res_2_=res(128)
        self.res_3_=res(256)
        self.res_4_=res(512)



        self.up2_conv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.up2_conv1_later = Conv_block(1024, 512)
        self.up2_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up2_conv2_later = Conv_block(512, 256)
        self.up2_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up2_conv3_later = Conv_block(256, 128)
        self.up2_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up2_conv4_later = Conv_block(128, 64)

        
        self.up2_conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up2_conv5_later =  nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)
        )

        self.up2_conv1_ = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.up2_conv1_later_ = Conv_block(1024, 512)
        self.up2_conv2_ = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up2_conv2_later_ = Conv_block(512, 256)
        self.up2_conv3_ = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up2_conv3_later_ = Conv_block(256, 128)
        self.up2_conv4_ = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up2_conv4_later_ = Conv_block(128, 64)

        
        
        self.up2_conv5_ = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up2_conv5_later_ =  nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)
        )
        
        

        self.out2_conv = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1, stride=1)

        self.relu=nn.ReLU(inplace=True)
        self.tanh=nn.Tanh()
        

    def forward(self,x):


        features_stack_2 = []
        x_lose=self.down2_conv0(x)
        x = self.down2_conv1(x_lose)
        
        features_stack_2.append(self.res_1_(x))
        
        x = self.down2_conv2(x)
        features_stack_2.append(self.res_2_(x))
        

        x = self.down2_conv3(x)
        
        features_stack_2.append(self.res_3_(x))
        

        x = self.down2_conv4(x)

        features_stack_2.append(self.res_4_(x))

        x = self.down2_conv5(x)
        
        x_a = (x).shape[1]

        x1=x[:,x_a//2:,:,:].clone()
        x=x[:,:x_a//2,:,:].clone()
        
        feat_cpy=features_stack_2.copy()
        x = self.up2_conv1(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv1_later(x)

        x = self.up2_conv2(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv2_later(x)

        x = self.up2_conv3(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv3_later(x)

        x = self.up2_conv4(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv4_later(x)
        

        x = self.up2_conv5(x)
        x = torch.cat((x,x_lose), dim=1)
        x = self.up2_conv5_later(x)

        out2_a = self.tanh((x))

        

        features_stack_2=feat_cpy
        x = self.up2_conv1_(x1)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv1_later_(x)

        x = self.up2_conv2_(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv2_later_(x)

        x = self.up2_conv3_(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv3_later_(x)

        x = self.up2_conv4_(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv4_later_(x)
        
        

        x = self.up2_conv5_(x)
        
        x = torch.cat((x,x_lose), dim=1)
        x = self.up2_conv5_later_(x)

        out2_b = self.tanh((x))


        out2=torch.cat([out2_a.permute(0,2,3,1),out2_b.permute(0,2,3,1)],dim=3)

        return out2