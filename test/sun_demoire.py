# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
#from modules import MLB,CARB,upsample_block

class Down0(nn.Module):
    def __init__(self):
        super(Down0, self).__init__()
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.down = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.down(x))
        LR = self.convt_R1(out)
        return LR

class Down1(nn.Module):
    def __init__(self):
        super(Down1, self).__init__()
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.down(x))
        LR_2x = self.convt_R1(out)
        return LR_2x

class Down2(nn.Module):
    def __init__(self):
        super(Down2, self).__init__()
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.down = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.down(x))
        LR_2x = self.convt_R1(out)
        return LR_2x


class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        return out

class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.u1 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.u1(x))
        clean = self.conv(out)
        return clean

class Branch3(nn.Module):
    def __init__(self):
        super(Branch3, self).__init__()
        self.u1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.u2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.u1(x))
        out = self.relu2(self.u2(out))
        clean = self.conv(out)
        return clean

class Branch4(nn.Module):
    def __init__(self):
        super(Branch4, self).__init__()
        self.u1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.u2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.u3 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.u1(x))
        out = self.relu2(self.u2(out))
        out = self.relu3(self.u3(out))
        clean = self.conv(out)
        return clean

class Branch5(nn.Module):
    def __init__(self):
        super(Branch5, self).__init__()
        self.u1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.u2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.u3 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.u4 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.relu4 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu1(self.u1(x))
        out = self.relu2(self.u2(out))
        out = self.relu3(self.u3(out))
        out = self.relu4(self.u4(out))
        clean = self.conv(out)
        return clean

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.relu = nn.PReLU()
        # 下采样
        self.down0 = Down0()
        self.down1 = Down1()
        self.down2 = Down2()
        self.down3 = Down2()
        self.down4 = Down2()
        # Branches
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()
        self.branch4 = Branch4()
        self.branch5 = Branch5()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # out = self.relu(self.conv_input(x))
        feat_down1 = self.down0(x)
        b1 = self.branch1(feat_down1)
        # print('b1',b1.shape)
        #---------
        feat_down2 = self.down1(feat_down1)
        b2 = self.branch2(feat_down2)
        # print('b2',b2.shape)
        #---------
        feat_down3 = self.down2(feat_down2)
        b3 = self.branch3(feat_down3)
        # print('b3',b3.shape)
        #---------
        feat_down4 = self.down3(feat_down3)
        b4 = self.branch4(feat_down4)
        # print('b4',b4.shape)
        #---------
        feat_down5 = self.down4(feat_down4)
        b5 = self.branch5(feat_down5)
        # print('b5',b5.shape)
        # clean = x + b1 + b2 + b3 + b4
        clean = b1 + b2 + b3 + b4 + b5
        # clean = self.convt_shape1(combine)

        return clean
