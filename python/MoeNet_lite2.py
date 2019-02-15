import torch
import torch.nn as nn

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

    res1 = self.ures1(convt_F13)
    im1 = self.uim1(out)
    u11 = self.convt_R1(res1)
    u12 = self.convt_I1(im1)
    HR = u11+u12

    return [HR]
