import torch
import torch.nn as nn
from functools import reduce

Conv3x3 = lambda channelIn, channelOut, stride=1: nn.Conv2d(in_channels=channelIn, out_channels=channelOut, kernel_size=3, stride=stride, padding=1, bias=False)
ConvTranspose2d4x4s2 = lambda channelIn, channelOut: nn.ConvTranspose2d(in_channels=channelIn, out_channels=channelOut, kernel_size=4, stride=2, padding=1)

class Down(nn.Module):
  def __init__(self, cin, cm, cout):
    super(Down, self).__init__()
    stride = 2 if cin == cm else 1
    self.relu = nn.PReLU()
    self.convt_R1 = Conv3x3(cm, cout)
    self.down = Conv3x3(cin, cm, stride)

  def forward(self, x):
    out = self.relu(self.down(x))
    LR = self.convt_R1(out)
    return LR

Branch = lambda *l: nn.Sequential(*sum(([ConvTranspose2d4x4s2(c1, c2), nn.PReLU()] for c1, c2 in zip(l, list(l[1:]) + [32])), []), Conv3x3(32, 3))
gNet = lambda f, acc, feat: (acc + f(feat), feat)
fNet = lambda acc, cur: gNet(cur[1], acc[0], cur[0](acc[1]))
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.downs = nn.ModuleList((Down(3, 32, 32), Down(32, 32, 64), Down(64, 64, 64), Down(64, 64, 64), Down(64, 64, 64)))
    branch1 = nn.Sequential(Conv3x3(32, 3), nn.PReLU())
    self.branches = nn.ModuleList((branch1, Branch(64), Branch(64, 64), Branch(64, 64, 32), Branch(64, 64, 32, 32)))
    self.ms = tuple(zip(self.downs, self.branches))

  def forward(self, x):
    return reduce(fNet, self.ms, (0, x))[:1]
