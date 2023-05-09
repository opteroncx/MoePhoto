import torch
import torch.nn as nn
import torch.nn.functional as F

from models import LayerNorm2d

class SimpleGate(nn.Module):
  def forward(self, x):
    x1, x2 = x.chunk(2, dim=1)
    return x1 * x2

class NAFBlock(nn.Module):
  def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
    super().__init__()
    dw_channel = c * DW_Expand
    self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                            bias=True)
    self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    # Simplified Channel Attention
    self.sca = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                groups=1, bias=True),
    )

    self.sg = SimpleGate()

    ffn_channel = FFN_Expand * c
    self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
    self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    self.norm1 = LayerNorm2d(c)
    self.norm2 = LayerNorm2d(c)

    self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

  def forward(self, inp):
    x = self.norm1(inp)

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.sg(x)
    x = x * self.sca(x)
    x = self.conv3(x)

    x = self.dropout(x)

    y = inp + x * self.beta

    x = self.conv4(self.norm2(y))
    x = self.sg(x)
    x = self.conv5(x)

    x = self.dropout(x)

    return y + x * self.gamma

class UNetLayer(nn.Module):
  def __init__(self, encoder, down, up, decoder, bottom=None):
    super(UNetLayer, self).__init__()
    self.encoder = encoder
    self.down = down
    self.up = up
    self.decoder = decoder
    self.bottom = (lambda x: bottom(x)) if bottom else (lambda x: x)
  def forward(self, x):
    x1 = self.encoder(x)
    x = self.up(self.bottom(self.down(x1)))
    return self.decoder(x + x1)

# align = 16 or 2^len(enc_blk_nums)
class NAFNet(nn.Module):

  def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
    super().__init__()

    self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=True)
    self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=True)

    self.layers = nn.ModuleList()

    chan = width << len(enc_blk_nums)
    m = nn.Sequential(
        *(NAFBlock(chan) for _ in range(middle_blk_num))
      )
    self.layers.append(m)
    for nEnc, nDec in zip(enc_blk_nums[::-1], dec_blk_nums):
      chan = chan >> 1
      m = UNetLayer(
        nn.Sequential(
          *(NAFBlock(chan) for _ in range(nEnc))
        ),
        nn.Conv2d(chan, chan << 1, 2, 2),
        nn.Sequential(
          nn.Conv2d(chan << 1, chan << 2, 1, bias=False),
          nn.PixelShuffle(2)
        ),
        nn.Sequential(
          *(NAFBlock(chan) for _ in range(nDec))
        ),
        m
      )
      self.layers.insert(0, m)

  def forward(self, inp):
    x = self.intro(inp)
    x = self.layers[0](x)
    x = self.ending(x)
    return x + inp