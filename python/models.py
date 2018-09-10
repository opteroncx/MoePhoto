# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class upsample_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample_block,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
    def forward(self,x):
        return self.prelu(self.shuffler(self.conv(x)))

class upsample_block3(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample_block3,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(3)
        self.prelu = nn.PReLU()
    def forward(self,x):
        return self.prelu(self.shuffler(self.conv(x)))

# Automatical residual scaling block (ARSB) architecture
class ARSB(nn.Module):
  def __init__(self, nChannels):
    super(ARSB, self).__init__()
    # reshape the channel size
    self.conv_1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
    self.conv_2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
    self.relu = nn.PReLU()
    self.scale = ScaleLayer()
  def forward(self, x):
    out = self.relu(self.conv_1(x))
    out = self.conv_2(out)
    out = self.scale(out) + x
    return out

class NetDN(nn.Module):
    def __init__(self):
        super(NetDN, self).__init__()

        filters = 48
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

        self.convt_R1 = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # add multi supervise
        self.convt_F11 = ARSB(filters)
        self.convt_F12 = ARSB(filters)
        self.convt_F13 = ARSB(filters)
        self.convt_F14 = ARSB(filters)
        self.convt_F15 = ARSB(filters)
        self.convt_F16 = ARSB(filters)
        # self.convt_F17 = ARSB(filters)
        self.convt_shape1 = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)


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
        convt_F14 = self.convt_F14(convt_F13)
        convt_F15 = self.convt_F15(convt_F14)
        convt_F16 = self.convt_F16(convt_F15)

        # multi supervise
        convt_F = [convt_F11,convt_F12,convt_F13,convt_F14,convt_F15,convt_F16]

        u2 = self.convt_shape1(out)

        HR = []

        for i in range(len(convt_F)):
            convt_R1 = self.convt_R1(convt_F[i])
            tmp = u2 + convt_R1
            HR.append(tmp)

        return HR

class ScaleLayer(nn.Module):

   def __init__(self, init_value=0.25):
       super(ScaleLayer,self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale

class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4),1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)

class Net2x(nn.Module):
    def __init__(self):
        super(Net2x, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.ures1 = upsample_block(64,256)
        self.ures2 = upsample_block(64,256)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # add multi supervise
        self.convt_F11 = ARSB(64)
        self.convt_F12 = ARSB(64)
        self.convt_F13 = ARSB(64)
        self.convt_F14 = ARSB(64)
        self.convt_F15 = ARSB(64)
        self.convt_F16 = ARSB(64)

        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

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
        convt_F14 = self.convt_F14(convt_F13)
        convt_F15 = self.convt_F15(convt_F14)
        convt_F16 = self.convt_F16(convt_F15)
        # multi supervise
        convt_F = [convt_F11,convt_F12,convt_F13,convt_F14,convt_F15,convt_F16]

        u1 = self.u1(out)
        u2 = self.convt_shape1(u1)

        HR = []

        for i in range(len(convt_F)):
            res1 = self.ures1(convt_F[i])
            convt_R1 = self.convt_R1(res1)
            tmp = u2 + convt_R1
            HR.append(tmp)

        return HR

class Net3x(nn.Module):
    def __init__(self):
        super(Net3x, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.u1 = upsample_block3(64,576)
        self.ures1 = upsample_block3(64,576)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # add multi supervise
        self.convt_F11 = ARSB(64)
        self.convt_F12 = ARSB(64)
        self.convt_F13 = ARSB(64)
        self.convt_F14 = ARSB(64)
        self.convt_F15 = ARSB(64)
        self.convt_F16 = ARSB(64)

        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

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
        convt_F14 = self.convt_F14(convt_F13)
        convt_F15 = self.convt_F15(convt_F14)
        convt_F16 = self.convt_F16(convt_F15)
        # multi supervise
        convt_F = [convt_F11,convt_F12,convt_F13,convt_F14,convt_F15,convt_F16]

        u1 = self.u1(out)
        u2 = self.convt_shape1(u1)

        HR = []

        for i in range(len(convt_F)):
            res1 = self.ures1(convt_F[i])
            convt_R1 = self.convt_R1(res1)
            tmp = u2 + convt_R1
            HR.append(tmp)

        return HR

class Net4x(nn.Module):
    def __init__(self):
        super(Net4x, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.ures1 = upsample_block(64,256)
        self.ures2 = upsample_block(64,256)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # add multi supervise
        self.convt_F11 = ARSB(64)
        self.convt_F12 = ARSB(64)
        self.convt_F13 = ARSB(64)
        self.convt_F14 = ARSB(64)
        self.convt_F15 = ARSB(64)
        self.convt_F16 = ARSB(64)

        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

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
        convt_F14 = self.convt_F14(convt_F13)
        convt_F15 = self.convt_F15(convt_F14)
        convt_F16 = self.convt_F16(convt_F15)

        # multi supervise
        convt_F = [convt_F11,convt_F12,convt_F13,convt_F14,convt_F15,convt_F16]

        u1 = self.u1(out)
        u2 = self.u2(u1)
        u2 = self.convt_shape1(u2)

        HR = []

        for i in range(len(convt_F)):
            res1 = self.ures1(convt_F[i])
            res2 = self.ures2(res1)
            convt_R1 = self.convt_R1(res2)
            tmp = u2 + convt_R1
            HR.append(tmp)

        return HR