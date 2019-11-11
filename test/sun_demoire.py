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

class Branch1(nn.Module):
  def __init__(self):
    super(Branch1, self).__init__()
    self.conv_input = Conv3x3(32, 3)
    self.relu = nn.PReLU()

  def forward(self, x):
    out = self.relu(self.conv_input(x))
    return out

class Branch2(nn.Module):
  def __init__(self):
    super(Branch2, self).__init__()
    self.conv = Conv3x3(32, 3)
    self.relu = nn.PReLU()
    self.u1 = ConvTranspose2d4x4s2(64, 32)

  def forward(self, x):
    out = self.relu(self.u1(x))
    clean = self.conv(out)
    return clean

class Branch3(nn.Module):
  def __init__(self):
    super(Branch3, self).__init__()
    self.u1 = ConvTranspose2d4x4s2(64, 64)
    self.u2 = ConvTranspose2d4x4s2(64, 32)
    self.conv = Conv3x3(32, 3)
    self.relu1 = nn.PReLU()
    self.relu2 = nn.PReLU()

  def forward(self, x):
    out = self.relu1(self.u1(x))
    out = self.relu2(self.u2(out))
    clean = self.conv(out)
    return clean

class Branch4(nn.Module):
  def __init__(self):
    super(Branch4, self).__init__()
    self.u1 = ConvTranspose2d4x4s2(64, 64)
    self.u2 = ConvTranspose2d4x4s2(64, 32)
    self.u3 = ConvTranspose2d4x4s2(32, 32)
    self.conv = Conv3x3(32, 3)
    self.relu1 = nn.PReLU()
    self.relu2 = nn.PReLU()
    self.relu3 = nn.PReLU()

  def forward(self, x):
    out = self.relu1(self.u1(x))
    out = self.relu2(self.u2(out))
    out = self.relu3(self.u3(out))
    clean = self.conv(out)
    return clean

class Branch5(nn.Module):
  def __init__(self):
    super(Branch5, self).__init__()
    self.u1 = ConvTranspose2d4x4s2(64, 64)
    self.u2 = ConvTranspose2d4x4s2(64, 32)
    self.u3 = ConvTranspose2d4x4s2(32, 32)
    self.u4 = ConvTranspose2d4x4s2(32, 32)
    self.conv = Conv3x3(32, 3)
    self.relu1 = nn.PReLU()
    self.relu2 = nn.PReLU()
    self.relu3 = nn.PReLU()
    self.relu4 = nn.PReLU()

  def forward(self, x):
    out = self.relu1(self.u1(x))
    out = self.relu2(self.u2(out))
    out = self.relu3(self.u3(out))
    out = self.relu4(self.u4(out))
    clean = self.conv(out)
    return clean

def Branch(*l):
  return nn.Sequential(ConvTranspose2d4x4s2(64, 64), nn.PReLU(), ConvTranspose2d4x4s2(64, 32), nn.PReLU(), ConvTranspose2d4x4s2(32, 32), nn.PReLU(), ConvTranspose2d4x4s2(32, 32), nn.PReLU(), Conv3x3(32, 3))

gNet = lambda f, acc, feat: (acc + f(feat), feat)
fNet = lambda acc, cur: gNet(cur[1], acc[0], cur[0](acc[1]))
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.downs = nn.ModuleList((Down(3, 32, 32), Down(32, 32, 64), Down(64, 64, 64), Down(64, 64, 64), Down(64, 64, 64)))
    self.branches = nn.ModuleList((Branch1(), Branch2(), Branch3(), Branch4(), Branch5()))
    self.ms = tuple(zip(self.downs, self.branches))

  def forward(self, x):
    return reduce(fNet, self.ms, (0, x))[:1]
