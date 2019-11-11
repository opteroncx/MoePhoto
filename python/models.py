# -*- coding:utf-8 -*-
# pylint: disable=E1101
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def initParameters(model):
  for i, convt in enumerate(model.convt_F):
    model.add_module('convt_F{}'.format(i + 1), convt)
  initConvParameters(model)

def initConvParameters(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      m.weight.data.normal_(0, math.sqrt(2. / n))
      if m.bias is not None:
        m.bias.data.zero_()

def genUpsampleBlock(r):
  class block(nn.Module):
    def __init__(self,in_channels,out_channels):
      super(block,self).__init__()
      self.conv = nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
      self.shuffler = nn.PixelShuffle(r)
      self.prelu = nn.PReLU()
    def forward(self,x):
      return self.prelu(self.shuffler(self.conv(x)))
  return block

upsample_block = genUpsampleBlock(2)
upsample_block3 = genUpsampleBlock(3)

Conv3x3 = lambda channelIn, channelOut, stride=1: nn.Conv2d(in_channels=channelIn, out_channels=channelOut, kernel_size=3, stride=stride, padding=1, bias=False)

def multiConvt(model, convt_R1, x, u):
  HR = []
  for convt in model.convt_F:
    x = convt(x)
    # add multi supervise
    if model.training:
      convt_R1 = convt_R1(x)
      HR.append(u + convt_R1)
  if not model.training:
    convt_R1 = convt_R1(x)
    HR.append(u + convt_R1)
  return HR

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

class MyNet(nn.Module):
  def __init__(self, filters = 64):
    super(MyNet, self).__init__()

    self.conv_input = Conv3x3(1, filters)
    self.conv_input2 = Conv3x3(filters, filters)
    self.relu = nn.PReLU()
    self.convt_F = [ARSB(filters) for _ in range(6)]

  def forward(self, x):
    out = self.relu(self.conv_input(x))
    conv1 = self.conv_input2(out)

    u = self.u(out)

    return multiConvt(self, self.convt_R1, conv1, u)

class Net2x(MyNet):
  def __init__(self):
    super(Net2x, self).__init__()
    self.u, self.convt_R1 = (nn.Sequential(
      upsample_block(64,256),
      Conv3x3(64,1)
    ) for _ in range(2))

    initParameters(self)

class Net3x(MyNet):
  def __init__(self):
    super(Net3x, self).__init__()
    self.u, self.convt_R1 = (nn.Sequential(
      upsample_block3(64,576),
      Conv3x3(64,1)
    ) for _ in range(2))

    initParameters(self)

class Net4x(MyNet):
  def __init__(self):
    super(Net4x, self).__init__()
    self.u, self.convt_R1 = (nn.Sequential(
      upsample_block(64,256),
      upsample_block(64,256),
      Conv3x3(64,1)
    ) for _ in range(2))

    initParameters(self)

# denoise models

class NetDN(MyNet):
  def __init__(self):
    filters = 48
    super(NetDN, self).__init__(filters)
    self.convt_R1, self.u = (Conv3x3(filters, 1) for _ in range(2))

    initParameters(self)

class _Conv_Block(nn.Module):
  def __init__(self):
    super(_Conv_Block, self).__init__()

    """unused:
    self.upsample = nn.Sequential(
      nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
    )
    """
    self.rblock = nn.Sequential(
      Conv3x3(64, 64),
      nn.LeakyReLU(0.2, inplace=True),
      Conv3x3(64, 64),
      nn.LeakyReLU(0.2, inplace=True),
      Conv3x3(64, 64 * 4),
    )
    self.trans = nn.Sequential(
      nn.Conv2d(in_channels=64 * 4, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
    )
    self.relu = nn.LeakyReLU(0.2, inplace=True)

    self.global_pool = nn.AdaptiveAvgPool2d(1)
    self.conv_down = nn.Conv2d(
      64 * 4, 64 // 4, kernel_size=1, bias=False)
    self.conv_up = nn.Conv2d(
      64 // 4, 64 * 4, kernel_size=1, bias=False)
    self.sig = nn.Sigmoid()

  def resBlock1(self, x):
    out=self.rblock(x)
    out1 = self.global_pool(out)
    out1 = self.conv_down(out1)
    out1 = self.relu(out1)
    out1 = self.conv_up(out1)
    out1 = self.sig(out1)
    out = out * out1
    out = self.trans(out)
    out = x + out
    return out

  def forward(self, x):
    return self.resBlock1(x)

def make_layer(block, num_of_layer):
  return nn.Sequential(*(block() for _ in range(num_of_layer)))

class SEDN(nn.Module):
  def __init__(self):
    super(SEDN, self).__init__()

    self.conv_input = Conv3x3(1, 64)
    self.relu = nn.LeakyReLU(0.2, inplace=True)
    self.convt_R1 = Conv3x3(64, 1)
    self.convt_F = [make_layer(_Conv_Block,16)]

    initParameters(self)

  def forward(self, x):
    out = self.relu(self.conv_input(x))
    return multiConvt(self, self.convt_R1, out, x)
