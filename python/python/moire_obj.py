# -*- coding:utf-8 -*-
# pylint: disable=E1101
import torch
import torch.nn as nn
from models import upsample_block, ScaleLayer, initConvParameters, Conv3x3, namedSequential, Residual, toModule, Space_attention, CARB, CAT, FRM, eF
from imageProcess import identity

RK3Trans = lambda n, k, bias: nn.Sequential(nn.PReLU(n, 0.25), nn.Conv2d(n, n, k, padding=(k // 2), bias=bias))
class RK3(nn.Module):
  def __init__(self, n_feats=64, kernel_size=3, bias=True):
    super(RK3, self).__init__()

    self.ms = nn.ModuleList([RK3Trans(n_feats, kernel_size, bias) for _ in range(3)])
    self.scale = nn.ModuleList([ScaleLayer(s) for s in (0.5, 2.0, -1.0, 2 / 3, 1 / 6)])

  def forward(self, x):
    k1 = self.ms[0](x)
    yn_1 = self.scale[0](k1) + x
    k2 = self.ms[1](yn_1)
    yn_2 = self.scale[1](k2) + self.scale[2](k1) + x
    k3 = self.ms[2](yn_2)
    return self.scale[3](k2) + self.scale[4](k3 + k1) + x

Down2 = lambda c_in, c_out: namedSequential(
  ('conv_input', Conv3x3(c_in, 32)),
  ('relu', nn.PReLU()),
  ('down', Conv3x3(32, 32, stride=2)),
  ('convt_R1', Conv3x3(32, c_out)),
  ('block', CARB(64)))

class Branch(nn.Module):
  def __init__(self, deepFs, combine=None, cat=True, u2=identity, in_channels=64):
    super(Branch, self).__init__()
    self.inputF = namedSequential(('conv_input', Conv3x3(in_channels, 64)), ('relu', nn.PReLU()))
    if cat:
      deepFs = [CAT(128)] + list(deepFs)
      self.shallowF = nn.Sequential(*(CARB(64) for _ in range(5)))
    else:
      self.shallowF = None
    self.deepF = nn.Sequential(*deepFs)
    self.combineF = namedSequential(*combine) if combine else None
    self.u2 = u2

    initConvParameters(self)

  def forward(self, x, t=None):
    out = self.inputF(x)
    if self.shallowF:
      b = t[1] # drop upsampled branch
      shallow_ft = self.shallowF(out)
      fu = torch.cat((shallow_ft, b), 1)
    else:
      fu = out
    deep = self.deepF(fu)
    combine = self.combineF(out + deep) if self.combineF else deep # pylint: disable=not-callable
    return self.u2(combine), combine

Branch1 = lambda: Branch((*(CARB(64) for _ in range(7)), RK3(), RK3()), in_channels=3)
Branch2 = lambda: Branch(
  (Space_attention(64, 64, 1, 1, 0, 1), *(CARB(64) for _ in range(7)), RK3(), RK3()),
  (('SA2', Space_attention(64, 64, 1, 1, 0, 1)), ('u1', upsample_block(64, 256))))
Branch3 = lambda: Branch(
  (*(CARB(64) for _ in range(7)), RK3(), RK3(), RK3()),
  (('SA2', Space_attention(64, 64, 1, 1, 0, 1)), ('u1', upsample_block(64, 256))),
  False)# Not used for inference, upsample_block(64, 256))

To_clean_image = lambda ichannels=64: namedSequential(
  ('residual', Residual(namedSequential(
    ('gff', Conv3x3(ichannels, ichannels)), ('relu', nn.PReLU()), ('se', FRM(ichannels))))),
  ('conv_tail', Conv3x3(ichannels, ichannels)),
  ('relut', nn.PReLU()),
  ('conv_out', Conv3x3(ichannels, 3, bias=True)))

UNet = lambda down, up, dir: lambda f: toModule(lambda *fs: lambda x: eF(up)(eF(dir)(x), eF(f)(eF(down)(x))))(down, up, dir, f)
Net = lambda: namedSequential(
  ('U', UNet(('down2_1', Down2(3,64)), ('branch1', Branch1()), identity)(
    UNet(('down2_2', Down2(64,64)), ('branch2', Branch2()), ('SA2', Space_attention(64,64,1,1,0,1)))(
      namedSequential(('SA3', Space_attention(64,64,1,1,0,1)), ('branch3', Branch3()))))),
  ('extract0', toModule(lambda *_: lambda t: t[0])()),
  ('to_clean1', To_clean_image()))