"""
Official Code Implementation of:
"A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping"
Authors:    Hmrishav Bandyopadhyay,
            Tanmoy Dasgupta,
            Nibaran Das,
            Mita Nasipuri

Code: Hmrishav Bandyopadhyay

"""


import torch
from torch import nn

from unets.Punet import Punet
from unets.Sunet import Sunet

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()

        self.unet1=Punet()
        self.unet2=Sunet()

    def forward(self,x):
        x_1,edge_out=self.unet1(x)
        grid=self.unet2(x_1)

        return grid,edge_out

if __name__ == '__main__':
        

    x = torch.randn((1, 3, 64, 64))
    net = Net()
    out = net(x)
    print(out[0].size())
