# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class upsample_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample_block,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
    def forward(self,x):
        return self.prelu(self.shuffler(self.conv(x)))

class upsample_block_v1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample_block_v1,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,1,stride=1,padding=0)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
    def forward(self,x):
        return self.prelu(self.shuffler(self.conv(x)))

class FRM(nn.Module):
    '''The feature recalibration module'''
    def __init__(self, channel, reduction=16):
        super(FRM, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class LB(nn.Module):
  def __init__(self, inChannels,outChannels):
    super(LB, self).__init__()
    # reshape the channel size
    self.conv_1 = nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1, bias=False)
    self.conv_2 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False)
    self.relu = nn.PReLU()
    self.se = FRM(channel=48,reduction=16)
  def forward(self, x):
    out = self.relu(self.conv_1(x))
    out = self.conv_2(out)
    out = self.se(out) + x
    return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.PReLU()
        self.ures1 = upsample_block_v1(48,192)
        self.uim1 = upsample_block_v1(48,192)
        self.ures2 = upsample_block_v1(48,192)
        self.uim2 = upsample_block_v1(48,192)
        self.ures3 = upsample_block_v1(48,192)
        self.uim3 = upsample_block_v1(48,192)
        self.convt_R1 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.convt_I1 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        # add multi supervise
        self.convt_F11 = LB(48,48)
        self.convt_F12 = LB(48,48)
        self.convt_F13 = LB(48,48)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        conv1 = self.conv_input2(out)

        convt_F11 = self.convt_F11(conv1)
        convt_F12 = self.convt_F12(convt_F11)
        convt_F13 = self.convt_F13(convt_F12)

        res1 = self.ures1(convt_F13)
        res2 = self.ures2(res1)
        res3 = self.ures3(res2)
        im1 = self.uim1(out)
        im2 = self.uim2(im1)
        im3 = self.uim3(im2)
        u11 = self.convt_R1(res3)
        u12 = self.convt_I1(im3)
        HR = u11+u12

        return HR

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error)
        return loss
