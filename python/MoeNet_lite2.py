import torch
import torch.nn as nn
from imageProcess import apply, reduce
from models import FRM

upsample_block_v1 = lambda in_channels, out_channels:\
  nn.Sequential(nn.Conv2d(in_channels,out_channels,1,stride=1,padding=0), nn.PixelShuffle(2), nn.PReLU())

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
  def __init__(self, upscale=2):
    super(Net, self).__init__()
    self.upscale = upscale
    l = int(upscale).bit_length() - 1

    self.conv_input = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv_input2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
    self.relu = nn.PReLU()
    self.ures = nn.ModuleList([upsample_block_v1(48,192) for _ in range(l)])
    self.uim = nn.ModuleList([upsample_block_v1(48,192) for _ in range(l)])
    self.convt_R1 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
    self.convt_I1 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
    # add multi supervise
    self.convt_F11 = LB(48,48)
    self.convt_F12 = LB(48,48)
    self.convt_F13 = LB(48,48)

  def forward(self, x):
    out = self.relu(self.conv_input(x))
    conv1 = self.conv_input2(out)

    convt_F11 = self.convt_F11(conv1)
    convt_F12 = self.convt_F12(convt_F11)
    convt_F13 = self.convt_F13(convt_F12)

    res = reduce(apply, self.ures, convt_F13)
    im = reduce(apply, self.uim, out)
    u11 = self.convt_R1(res)
    u12 = self.convt_I1(im)
    HR = u11+u12

    return [HR]
