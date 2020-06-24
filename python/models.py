# -*- coding:utf-8 -*-
# pylint: disable=E1101
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from imageProcess import apply, reduce, identity
from collections import OrderedDict

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

genUpsampleBlock = lambda r: lambda in_channels, out_channels:\
  nn.Sequential(Conv3x3(in_channels, out_channels, bias=True), nn.PixelShuffle(r), nn.PReLU())

upsample_block = genUpsampleBlock(2)
upsample_block3 = genUpsampleBlock(3)

Conv3x3 = lambda channelIn, channelOut, stride=1, bias=False:\
  nn.Conv2d(in_channels=channelIn, out_channels=channelOut, kernel_size=3, stride=stride, padding=1, bias=bias)

residual = lambda f, x, u: u + f(x)
appendApply = lambda a, f: a + [f(a[-1])]

multiConvt = lambda model, convt_R1, x, u:\
  [residual(convt_R1, y, u) for y in reduce(appendApply, model.convt_F, [x])[1:]]\
  if model.training else [residual(convt_R1, reduce(apply, model.convt_F, x), u)]

namedSequential = lambda *args: nn.Sequential(OrderedDict(args))
isModule = lambda m: isinstance(m, nn.Module)
_addModule = lambda model, name, m: model.add_module(name, m) if isModule(m) else None
addModule = lambda model: lambda t: _addModule(model, *((t[1][0], t[1][1]) if type(t[1]) is tuple else (str(t[0]), t[1])))
addModules = lambda model, ms: tuple(map(addModule(model), enumerate(ms)))
eF = lambda t: t[1] if type(t) is tuple else t
extractFuncs = lambda args: map(eF, args)

def toModule(f):
  class M(nn.Module):
    def __init__(self, *fs):
      super(M, self).__init__()
      addModules(self, fs)
      self.f = f(*fs)

    def forward(self, *args): return self.f(*args)
  return M

Residual = toModule(lambda *fs: lambda x: sum(f(x) for f in extractFuncs(fs)) + x)

class ScaleLayer(nn.Module):

   def __init__(self, init_value=0.25):
     super(ScaleLayer,self).__init__()
     self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
     return input * self.scale

# Automatical residual scaling block (ARSB) architecture
ARSB = lambda nChannels: Residual(namedSequential(
  ('conv_1', Conv3x3(nChannels, nChannels)),
  ('relu', nn.PReLU()),
  ('conv_2', Conv3x3(nChannels, nChannels)),
  ('scale', ScaleLayer())))

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

make_layer = lambda block, num_of_layer: nn.Sequential(*(block() for _ in range(num_of_layer)))

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

class Space_attention(torch.nn.Module):
  def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
    super(Space_attention, self).__init__()

    self.output_size = output_size
    self.stride = stride
    self.scale = scale

    self.K = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
    self.Q = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
    self.V = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
    self.pool = nn.MaxPool2d(kernel_size=scale + 2, stride=scale, padding=1) if stride > 1 else identity
    #self.bn = nn.BatchNorm2d(output_size)
    if kernel_size == 1:
      self.local_weight = nn.Conv2d(output_size, input_size, kernel_size, stride, padding, bias=True)
    else:
      self.local_weight = nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding, bias=True)

  def forward(self, x):
    batch_size, _, h, w = x.shape
    K = self.K(x)
    Q = self.Q(x)
    # Q = F.interpolate(Q, scale_factor=1 / self.scale, mode='bicubic')
    Q = self.pool(Q)
    V = self.V(x)
    # V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
    V = self.pool(V)
    V_reshape = V.view(batch_size, self.output_size, -1)
    V_reshape = V_reshape.permute(0, 2, 1)
    # if self.type == 'softmax':
    Q_reshape = Q.view(batch_size, self.output_size, -1)

    K_reshape = K.view(batch_size, self.output_size, -1)
    K_reshape = K_reshape.permute(0, 2, 1)

    KQ = torch.matmul(K_reshape, Q_reshape)
    attention = F.softmax(KQ, dim=-1)

    vector = torch.matmul(attention, V_reshape)
    vector_reshape = vector.permute(0, 2, 1).contiguous()
    O = vector_reshape.view(batch_size, self.output_size, h // self.stride, w // self.stride)
    W = self.local_weight(O)
    output = x + W
    return output

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

CARBF = lambda n, r:\
  Residual(namedSequential(('conv1', Conv3x3(n, n)), ('relu', nn.PReLU()), ('conv2', Conv3x3(n, n)), ('ca', FRM(n, r))))
CARB = lambda nChannels, reduction=16: nn.Sequential(*(CARBF(nChannels, reduction) for _ in range(2)))

CAT = lambda channel, reduction=16:\
  nn.Sequential(FRM(channel, reduction), nn.Conv2d(channel, channel//2, kernel_size=1, padding=0, bias=True))

fsND = ((nn.Conv1d, nn.MaxPool1d, nn.BatchNorm1d), (nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d), (nn.Conv3d, nn.MaxPool3d, nn.BatchNorm3d))
conv110 = lambda conv_nd: lambda in_channels, out_channels: conv_nd(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
seriresM = lambda f, m: nn.Sequential(f, m) if f and m else (f if f else m)

_nonlocalMean = lambda f: f / f.size(-1)
_nonlocalSoftmax = lambda f: F.softmax(f, dim=-1)
_nonlocalMul = lambda _, theta_x, phi_x: torch.matmul(theta_x, phi_x)
def _nonlocalConcat(self, theta_x, phi_x):
  h = theta_x.size(2)
  w = phi_x.size(3)
  theta_x = theta_x.repeat(1, 1, 1, w)
  phi_x = phi_x.repeat(1, 1, h, 1)
  concat_feature = torch.cat([theta_x, phi_x], dim=1)
  return self.concat_project(concat_feature).squeeze(1)

operation_function = {
  'embedded_gaussian': lambda *args: _nonlocalSoftmax(_nonlocalMul(*args)),
  'gaussian': lambda *args: _nonlocalSoftmax(_nonlocalMul(*args)),
  'dot_product': lambda *args: _nonlocalMean(_nonlocalMul(*args)),
  'concatenation': lambda *args: _nonlocalMean(_nonlocalConcat(*args))
}
class _NonLocalBlockND(nn.Module):
  def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
              sub_sample=True, bn_layer=True):
    super(_NonLocalBlockND, self).__init__()
    self.sub_sample = sub_sample
    self.in_channels = in_channels

    if inter_channels is None:
      inter_channels = max(1, in_channels // 2)
    self.inter_channels = inter_channels

    conv_nd, max_pool, bn = fsND(dimension - 1)
    convF = conv110(conv_nd)

    g = convF(in_channels, inter_channels)

    self.theta = None
    phi = None
    self.concat_project = None
    if mode in {'embedded_gaussian', 'dot_product', 'concatenation'}:
      self.theta = convF(in_channels, inter_channels)
      phi = convF(in_channels, inter_channels)

    if mode == 'concatenation':
      self.concat_project = nn.Sequential(
        nn.Conv2d(inter_channels * 2, 1, 1, 1, 0, bias=False),
        nn.ReLU()
      )

    self.operation_function = operation_function[mode]
    pool = lambda: max_pool(kernel_size=2) if sub_sample else lambda: None
    self.g = seriresM(g, pool())
    self.phi = seriresM(phi, pool())
    if not self.phi:
      self.phi = identity

    if bn_layer:
        self.W = nn.Sequential(
          convF(inter_channels, in_channels),
          bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
    else:
        self.W = convF(inter_channels, in_channels)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

  def forward(self, x):
    '''
    :param x: (b, c, t, h, w)
    :return:
    '''
    batch_size, _, *size = x.shape
    g_x = self.g(x)
    g_x = g_x.view(batch_size, self.inter_channels, -1)
    g_x = g_x.permute(0, 2, 1)
    theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
    theta_x = theta_x.permute(0, 2, 1)
    phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
    f_div_C = self.operation_function(self, theta_x, phi_x) # (batch, feature, feature)
    y = torch.matmul(f_div_C, g_x)
    y = y.permute(0, 2, 1).contiguous()
    y = y.view(batch_size, self.inter_channels, *size)
    W_y = self.W(y)
    z = W_y + x
    return z

NONLocalBlock2D = lambda in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True:\
  _NonLocalBlockND(in_channels, inter_channels, 2, mode, sub_sample, bn_layer)

## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
  def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
    super(Nonlocal_CA, self).__init__()
    # nonlocal module
    self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))
    self.sigmoid = nn.Sigmoid()
  def forward(self,x):
    ## divide feature map into 4 part
    batch_size,C,H,W = x.shape
    H1 = int(H / 2)
    W1 = int(W / 2)
    nonlocal_feat = torch.zeros_like(x)
    feat_sub_lu = x[:, :, :H1, :W1]
    feat_sub_ld = x[:, :, H1:, :W1]
    feat_sub_ru = x[:, :, :H1, W1:]
    feat_sub_rd = x[:, :, H1:, W1:]
    nonlocal_lu = self.non_local(feat_sub_lu)
    nonlocal_ld = self.non_local(feat_sub_ld)
    nonlocal_ru = self.non_local(feat_sub_ru)
    nonlocal_rd = self.non_local(feat_sub_rd)
    nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
    nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
    nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
    nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd
    return  nonlocal_feat