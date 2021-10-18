# Zhihang Zhong https://github.com/zzh-tech/ESTRNN
import torch
import torch.nn as nn
import torch.nn.functional as F

from imageProcess import ceilBy, StreamState, identity, doCrop
from runSlomo import getOptS, getOptP, makeStreamFunc
from progress import Node
from MPRNet import Residual

para = type('', (), {})()
para.future_frames = 2
para.past_frames = 2
para.activation = 'gelu'
para.n_blocks = 15
DS_ratio = 2
NumFeat = 16
RefTime = para.future_frames + 1 + para.past_frames
WindowSize = 1

conv1x1 = lambda in_channels, out_channels, stride=1: nn.Conv2d(
  in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
conv3x3 = lambda in_channels, out_channels, stride=1: nn.Conv2d(
  in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
conv5x5 = lambda in_channels, out_channels, stride=1: nn.Conv2d(
  in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

ActFuncs = dict(
  relu=nn.ReLU,
  relu6=nn.ReLU6,
  leakyrelu=(nn.LeakyReLU, 0.1),
  prelu=nn.PReLU,
  rrelu=(nn.RReLU, 0.1, 0.3),
  selu=nn.SELU,
  celu=nn.CELU,
  elu=nn.ELU,
  gelu=nn.GELU,
  tanh=nn.Tanh)
def actFunc(act):
  act = act.lower()
  if not act in ActFuncs:
    raise NotImplementedError
  act = ActFuncs[act]
  return act[0](*act[1:]) if type(act) == tuple else act()

class dense_layer(nn.Module):
  def __init__(self, in_channels, growthRate, activation='relu'):
    super(dense_layer, self).__init__()
    self.conv = conv3x3(in_channels, growthRate)
    self.act = actFunc(activation)

  def forward(self, x):
    out = self.act(self.conv(x))
    out = torch.cat((x, out), dim=1)
    return out

# Residual dense block
RDB = lambda in_channels, growthRate, num_layer, activation='relu': Residual(
  *(dense_layer(in_channels + i * growthRate, growthRate, activation) for i in range(num_layer)),
  conv1x1(in_channels + num_layer * growthRate, in_channels))

# Middle network of residual dense blocks
class RDNet(nn.Module):
  def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
    super(RDNet, self).__init__()
    self.RDBs = nn.ModuleList((RDB(in_channels, growthRate, num_layer, activation) for _ in range(num_blocks)))
    self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
    self.conv3x3 = conv3x3(in_channels, in_channels)

  def forward(self, x):
    out = []
    h = x
    for rdb in self.RDBs:
      h = rdb(h)
      out.append(h)
    out = torch.cat(out, dim=1)
    return self.conv3x3(self.conv1x1(out))

# DownSampling module
RDB_DS = lambda in_channels, growthRate, num_layer, activation='relu': nn.Sequential(
  RDB(in_channels, growthRate, num_layer, activation),
  conv5x5(in_channels, 2 * in_channels, stride=2)
)

# Global spatio-temporal attention module
class GSA(nn.Module):
  def __init__(self, para):
    super(GSA, self).__init__()
    self.center = para.past_frames
    self.num_ff = para.future_frames
    self.num_fb = para.past_frames
    ids = torch.arange(RefTime)
    self.ids = list(ids[ids != self.center])
    self.F_f = nn.Sequential(
      nn.Linear(2 * 5 * NumFeat, 4 * 5 * NumFeat),
      actFunc(para.activation),
      nn.Linear(4 * 5 * NumFeat, 2 * 5 * NumFeat),
      nn.Sigmoid()
    )
    # out channel: 10 * NumFeat
    self.F_p = nn.Sequential(
      conv1x1(2 * 5 * NumFeat, 4 * 5 * NumFeat),
      conv1x1(4 * 5 * NumFeat, 2 * 5 * NumFeat)
    )
    # condense layer
    self.condense = conv1x1(2 * 5 * NumFeat, 5 * NumFeat)
    # fusion layer
    self.fusion = conv1x1(RefTime * 5 * NumFeat, RefTime * 5 * NumFeat)

  def related(self, x):
    ref = x[:, self.center]
    return torch.stack([torch.cat([ref, x[:, i]], dim=1) for i in self.ids], dim=1), ref

  def forward(self, hs, weight, **_):
    bsz, _, c, h, w = hs.shape # bsz, RefTime, c, h, w
    weight, _ = self.related(weight) # bsz, RefTime - 1, 2 * c
    weight = self.F_f(weight).view(-1, c << 1, 1, 1) # bsz * (RefTime - 1), 2 * c, 1, 1
    cor, f_ref = self.related(hs) # (bsz, RefTime - 1, 2 * c, h, w), (bsz, c, h, w)
    cor = self.F_p(cor.view(-1, c << 1, h, w)) # bsz * (RefTime - 1), 2 * c, h, w
    cor = self.condense(weight * cor) # bsz * (RefTime - 1), c, h, w
    cor_l = torch.cat([cor.view(bsz, -1, h, w), f_ref], dim=1)

    return self.fusion(cor_l) # bsz, RefTime * c, h, w

# RDB-based RNN cell
class RDBCell(nn.Module):
  def __init__(self, para):
    super(RDBCell, self).__init__()
    activation = para.activation
    n_blocks = para.n_blocks
    self.F_B0 = conv5x5(3, NumFeat, stride=1)
    self.F_B1 = RDB_DS(in_channels=NumFeat, growthRate=NumFeat, num_layer=3, activation=activation)
    self.F_B2 = RDB_DS(in_channels=2 * NumFeat, growthRate=NumFeat * 3 // 2, num_layer=3,
                        activation=activation)
    self.F_R = RDNet(in_channels=(1 + 4) * NumFeat, growthRate=2 * NumFeat, num_layer=3,
                      num_blocks=n_blocks, activation=activation)
    # F_h: hidden state part
    self.F_h = nn.Sequential(
      conv3x3((1 + 4) * NumFeat, NumFeat),
      RDB(in_channels=NumFeat, growthRate=NumFeat, num_layer=3, activation=activation),
      conv3x3(NumFeat, NumFeat)
    )

  def forward(self, x, s_last):
    out = self.F_B0(x)
    out = self.F_B1(out)
    out = self.F_B2(out)
    out = torch.cat((out, s_last), dim=1)

    return self.F_R(out), self.F_h(out)

Reconstructor = lambda *_: nn.Sequential(
  nn.ConvTranspose2d(RefTime * 5 * NumFeat,
    2 * NumFeat, kernel_size=3, stride=2, padding=1, output_padding=1),
  nn.ConvTranspose2d(2 * NumFeat, NumFeat, kernel_size=3,
    stride=2, padding=1, output_padding=1),
  conv5x5(NumFeat, 3, stride=1)
)

def calcForward(opt, state, x, **_):
  if state.feat_hidden == None:
    height, width = x.shape[-2:] # (1, 3, height, width)
    s_height = height >> DS_ratio
    s_width = width >> DS_ratio
    state.feat_hidden = x.new_zeros(1, NumFeat, s_height, s_width)
  h, state.feat_hidden = opt.cell(x, state.feat_hidden)
  return h

pooling = lambda hs, **_: F.adaptive_avg_pool2d(hs, (1, 1)).view(*hs.shape[:2])

modelPaths = {
  '1ms8ms': './model/ESTRNN/ESTRNN_C80B15_BSD_1ms8ms.pth',
  '2ms16ms': './model/ESTRNN/ESTRNN_C80B15_BSD_2ms16ms.pth',
  '3ms24ms': './model/ESTRNN/ESTRNN_C80B15_BSD_3ms24ms.pth'
}
ramCoef = [.6 / x for x in (1., 10560.1, 6528., 1., 14536.8, 3228.6, 1., 7276.4, 2955.6)]
modules = dict(
  cell=dict(weight='cell', f=lambda *_: RDBCell(para)),
  fusion=dict(weight='fusion', outShape=(1, 5 * NumFeat * RefTime, 0.25, 0.25),
    streams=['fusionStream'], f=lambda *_: GSA(para)),
  recons=dict(weight='recons', outShape=(1, 3, 1, 1), streams=['out'],
    scale=4, f=Reconstructor)
)
getOpt = lambda option: getOptP(getOptS(modelPaths[option['model']], modules, ramCoef), bf=lambda *_: 1)

def initFunc(opt, x):
  *_, h, w = x.shape
  width = ceilBy(8)(w)
  height = ceilBy(8)(h)
  opt.pad = nn.ReflectionPad2d((0, width - w, 0, height - h))
  opt.unpad = lambda im: im[:, :h, :w]
  return height, width

# Efficient saptio-temporal recurrent neural network (ESTRNN, ECCV2020)
def doESTRNN(func, node, opt):
  nodes = [Node({'ESTRNN': key}) for key in ('forward', 'pooling', 'fusion', 'recons')]
  inp = StreamState(offload=False)
  forward = StreamState(offload=False)
  forward.feat_hidden = None
  StreamState.pipe(nodes[0].bindFunc(calcForward), [inp], [forward], args=[opt, forward])
  hs = StreamState(RefTime, reserve=1)
  inpW = StreamState()
  StreamState.pipe(identity, [forward], [hs, inpW])
  w = StreamState(RefTime, reserve=1)
  StreamState.pipe(nodes[1].bindFunc(pooling), [inpW], [w])
  fusion = StreamState(offload=False)
  opt.fusionStream = StreamState.pipe(nodes[2].bindFunc(opt.fusion), [hs, w], [fusion])
  recons = StreamState(store=False)
  opt.out = StreamState.pipe(nodes[3].bindFunc(doCrop), [fusion], [recons], args=[opt.recons])
  return makeStreamFunc(func, node, opt, nodes, 'ESTRNN', [hs, w], initFunc, inp.push)