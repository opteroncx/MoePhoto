# -*- coding:utf-8 -*-
import torch.nn as nn
from models import upsample_block, ScaleLayer, initConvParameters, Conv3x3, namedSequential, CARB, Nonlocal_CA
from imageProcess import identity, reduce

def calc_mean_std(feat):
  size = feat.size()
  assert (len(size) == 4)
  N, C = size[:2]
  featHW = feat.view(N, C, -1)
  feat_var = featHW.var(dim=2)
  feat_std = feat_var.sqrt().unsqueeze(-1).unsqueeze(-1)
  feat_mean = featHW.mean(dim=2).unsqueeze(-1).unsqueeze(-1)
  return feat_mean, feat_std

def din(content_feat, encode_feat, eps=1e-4):
  # eps is a small value added to the variance to avoid divide-by-zero.
  size = content_feat.size()
  content_mean, content_std = calc_mean_std(content_feat)
  encode_mean, encode_std = calc_mean_std(encode_feat)
  normalized_feat = (content_feat - content_mean) / (content_std + eps)
  return normalized_feat * encode_std + encode_mean

Down2 = lambda c_in, c_out: namedSequential(
  ('conv_input', Conv3x3(c_in, 32)),
  ('relu', nn.PReLU()),
  ('down', Conv3x3(32, 32, stride=2)),
  ('convt_R1', Conv3x3(32, c_out)))

Branch1 = lambda: namedSequential(
  ('conv_input', Conv3x3(3, 3)),
  ('relu', nn.PReLU()),
  ('conv_input2', Conv3x3(3, 3)))

couplePath = lambda feat, s: (din(feat, s), s)
forwardPath = lambda xs, fs: couplePath(fs[0](xs[0]), fs[1](xs[1]))
class Branch(nn.Module):
  def __init__(self, scaleLayers, strides, non_local=True):
    super(Branch, self).__init__()
    self.conv_input = Conv3x3(64, 64)
    self.relu = nn.PReLU()

    self.convt_F = nn.ModuleList(CARB(64) for _ in strides)
    # style encode
    self.s_conv = nn.ModuleList(Conv3x3(64, 64, stride=k) for k in strides)

    self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False) if non_local else identity

    self.u = nn.Sequential(*(upsample_block(64, 256) for _ in range(scaleLayers)))
    self.convt_shape1 = Conv3x3(64, 3)

    initConvParameters(self)

  def forward(self, x):
    out = self.relu(self.conv_input(x))
    convt_F = reduce(forwardPath, zip(self.convt_F, self.s_conv), (out, out))[0]
    combine = out + self.non_local(convt_F)
    #上采样
    up = self.u(combine)
    clean = self.convt_shape1(up)

    return clean

Branch2 = lambda: Branch(1, (1, 2, 2), False)
Branch3 = lambda: Branch(2, (1, 2, 1, 2))
Branch4 = lambda: Branch(3, (1, 2, 1, 2, 1, 2))
Branch5 = lambda: Branch(4, (1, 2, 1, 2, 1, 2, 1, 2))
Branch6 = lambda: Branch(5, (1, 1, 2, 1, 1, 2, 1, 1))

class Net(nn.Module):
  def __init__(self, layers=5):
    super(Net, self).__init__()
    self._down2 = nn.ModuleList([Down2(3,64)] + [Down2(64,64) for _ in range(layers - 2)])
    self.down2 = list(self._down2) + [identity]
    # Branches
    branches = (Branch1, Branch2, Branch3, Branch4, Branch5, Branch6)[:layers]
    self.branches = nn.ModuleList(f() for f in branches)
    self.scales = nn.ModuleList(ScaleLayer() for _ in range(layers))

    initConvParameters(self)

  def forward(self, x):
    return reduce((lambda a, fs: (fs[0](a[0]), fs[2](fs[1](a[0])) + a[1])), zip(self.down2, self.branches, self.scales), (x, 0))
