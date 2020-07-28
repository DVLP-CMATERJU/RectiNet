import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_block, self).__init__(**kwargs)

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


class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        

    def forward(self, input_features, gating_features):
        
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
  
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Conv2dPad(nn.Conv2d):
    def forward(self, input):
        return F.conv2d_same(input,self.weight,self.groups)



class res(nn.Module):
    def __init__(self,in_channels,**kwargs):
        super(res,self).__init__(**kwargs)
        self.c1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True))
        

    def forward(self,x):
        x1=self.c1(x)
        

        return x1



class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)

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

        

        self.res_1=res(64)
        self.res_2=res(128)
        self.res_3=res(256)
        self.res_4=res(512)

        self.res_1_=res(64)
        self.res_2_=res(128)
        self.res_3_=res(256)
        self.res_4_=res(512)

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


        self.out_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1)

        # second unet
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


        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)


        self.res1 = BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(32, 32)
        self.gate2 = GatedSpatialConv2d(16, 16)
        self.gate3 = GatedSpatialConv2d(8, 8)
        self.sigmoid=nn.Sigmoid()
        self.conv_bound=nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,padding=1)


        self.relu=nn.ReLU(inplace=True)
        self.tanh=nn.Tanh()
        self.out_conv_=nn.Sequential(nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=2,stride=2),
            nn.ReLU())


        
        

    def forward(self, x):

        features_stack = []
        features_stack_2 = []

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

        
        xa = F.interpolate(self.dsn1(x_1), x_size[2:],
                            mode='bilinear', align_corners=True)

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

        out_bound = cs

        out_bound=self.out_conv_(out_bound)

        out_bound_out= self.conv_bound(out_bound)

        out_bound_out= self.sigmoid(out_bound_out)



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

        

        x = torch.cat((x_lose, out1,out_bound), dim=1)
        
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

        out2 = self.tanh((x))

        x=x1

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

        out21 = self.tanh((x))


        out2=torch.cat([out2.permute(0,2,3,1),out21.permute(0,2,3,1)],dim=3)





        return out2,out_bound_out

