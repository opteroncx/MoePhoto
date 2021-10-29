# -*- coding:utf-8 -*-
import torch.nn as nn


class upsample_block_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_block_v1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              1, stride=1, padding=0)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
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


class ARSB(nn.Module):
    def __init__(self, nChannels):
        super(ARSB, self).__init__()
        # reshape the channel size
        self.conv_1 = nn.Conv2d(nChannels, nChannels,
                                kernel_size=3, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(nChannels, nChannels,
                                kernel_size=3, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.se = FRM(channel=48, reduction=16)

    def forward(self, x):
        out = self.relu(self.conv_1(x))
        out = self.conv_2(out)
        out = self.se(out) + x
        return out

def debug(x, name):
    from PIL import Image
    from imageProcess import toFloat, toOutput8
    x = x.squeeze(1)
    if x.shape[0] == 1:
        x = x.squeeze(0)
    Image.fromarray(toOutput8(toFloat(x))).save('./download/{}.png'.format(name))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(
            in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.ures1 = upsample_block_v1(48, 192)
        self.convt_R1 = nn.Conv2d(
            in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F11 = ARSB(48)
        self.convt_F12 = ARSB(48)
        self.convt_F13 = ARSB(48)

    def forward(self, x, base):
        out = self.relu(self.conv_input(x))
        conv1 = self.conv_input2(out)
        convt_F11 = self.convt_F11(conv1)
        convt_F12 = self.convt_F12(convt_F11)
        convt_F13 = self.convt_F13(convt_F12)
        res1 = self.ures1(convt_F13)
        convt_R1 = self.convt_R1(res1)
        HR = base + convt_R1

        return HR
