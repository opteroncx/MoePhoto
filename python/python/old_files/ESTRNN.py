# Zhihang Zhong https://github.com/zzh-tech/ESTRNN
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

def actFunc(act, *args, **kwargs):
  act = act.lower()
  if act == 'relu':
    return nn.ReLU()
  elif act == 'relu6':
    return nn.ReLU6()
  elif act == 'leakyrelu':
    return nn.LeakyReLU(0.1)
  elif act == 'prelu':
    return nn.PReLU()
  elif act == 'rrelu':
    return nn.RReLU(0.1, 0.3)
  elif act == 'selu':
    return nn.SELU()
  elif act == 'celu':
    return nn.CELU()
  elif act == 'elu':
    return nn.ELU()
  elif act == 'gelu':
    return nn.GELU()
  elif act == 'tanh':
    return nn.Tanh()
  else:
    raise NotImplementedError

# Dense layer
class dense_layer(nn.Module):
  def __init__(self, in_channels, growthRate, activation='relu'):
    super(dense_layer, self).__init__()
    self.conv = conv3x3(in_channels, growthRate)
    self.act = actFunc(activation)

  def forward(self, x):
    out = self.act(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


# Residual dense block
class RDB(nn.Module):
  def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
    super(RDB, self).__init__()
    in_channels_ = in_channels
    modules = []
    for i in range(num_layer):
      modules.append(dense_layer(in_channels_, growthRate, activation))
      in_channels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv1x1 = conv1x1(in_channels_, in_channels)

  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv1x1(out)
    out += x
    return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
  def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
    super(RDNet, self).__init__()
    self.num_blocks = num_blocks
    self.RDBs = nn.ModuleList()
    for i in range(num_blocks):
      self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation))
    self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
    self.conv3x3 = conv3x3(in_channels, in_channels)

  def forward(self, x):
    out = []
    h = x
    for i in range(self.num_blocks):
      h = self.RDBs[i](h)
      out.append(h)
    out = torch.cat(out, dim=1)
    out = self.conv1x1(out)
    out = self.conv3x3(out)
    return out


# DownSampling module
class RDB_DS(nn.Module):
  def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
    super(RDB_DS, self).__init__()
    self.rdb = RDB(in_channels, growthRate, num_layer, activation)
    self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

  def forward(self, x):
    # x: n,c,h,w
    x = self.rdb(x)
    out = self.down_sampling(x)

    return out


# Global spatio-temporal attention module
class GSA(nn.Module):
  def __init__(self, para):
    super(GSA, self).__init__()
    self.n_feats = para.n_features
    self.center = para.past_frames
    self.num_ff = para.future_frames
    self.num_fb = para.past_frames
    self.related_f = self.num_ff + 1 + self.num_fb
    self.F_f = nn.Sequential(
      nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
      actFunc(para.activation),
      nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
      nn.Sigmoid()
    )
    # out channel: 160
    self.F_p = nn.Sequential(
      conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
      conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
    )
    # condense layer
    self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
    # fusion layer
    self.fusion = conv1x1(self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))

  def forward(self, hs):
    # hs: [(n=4,c=80,h=64,w=64), ..., (n,c,h,w)]
    self.nframes = len(hs)
    f_ref = hs[self.center]
    cor_l = []
    for i in range(self.nframes):
      if i != self.center:
        cor = torch.cat([f_ref, hs[i]], dim=1)
        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)
        cor = self.F_p(cor)
        cor = self.condense(w * cor)
        cor_l.append(cor)
    cor_l.append(f_ref)
    out = self.fusion(torch.cat(cor_l, dim=1))

    return out


# RDB-based RNN cell
class RDBCell(nn.Module):
  def __init__(self, para):
    super(RDBCell, self).__init__()
    self.activation = para.activation
    self.n_feats = para.n_features
    self.n_blocks = para.n_blocks
    self.F_B0 = conv5x5(3, self.n_feats, stride=1)
    self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
    self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                        activation=self.activation)
    self.F_R = RDNet(in_channels=(1 + 4) * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                      num_blocks=self.n_blocks, activation=self.activation)  # in: 80
    # F_h: hidden state part
    self.F_h = nn.Sequential(
      conv3x3((1 + 4) * self.n_feats, self.n_feats),
      RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation),
      conv3x3(self.n_feats, self.n_feats)
    )

  def forward(self, x, s_last):
    out = self.F_B0(x)
    out = self.F_B1(out)
    out = self.F_B2(out)
    out = torch.cat([out, s_last], dim=1)
    out = self.F_R(out)
    s = self.F_h(out)

    return out, s


# Reconstructor
class Reconstructor(nn.Module):
  def __init__(self, para):
    super(Reconstructor, self).__init__()
    self.para = para
    self.num_ff = para.future_frames
    self.num_fb = para.past_frames
    self.related_f = self.num_ff + 1 + self.num_fb
    self.n_feats = para.n_features
    self.model = nn.Sequential(
      nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                          padding=1, output_padding=1),
      nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
      conv5x5(self.n_feats, 3, stride=1)
    )

  def forward(self, x):
    return self.model(x)

"""
self.parser.add_argument('--model', type=str, default='ESTRNN', help='type of model to construct')
self.parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
self.parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
self.parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
self.parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
self.parser.add_argument('--activation', type=str, default='gelu', help='activation function')
"""

class Model(nn.Module):
  """
  Efficient saptio-temporal recurrent neural network (ESTRNN, ECCV2020)
  """
  def __init__(self, para):
    super(Model, self).__init__()
    self.n_feats = para.n_features
    self.num_ff = para.future_frames
    self.num_fb = para.past_frames
    self.ds_ratio = 2
    self.cell = RDBCell(para)
    self.recons = Reconstructor(para)
    self.fusion = GSA(para)

  def forward(self, x):
    batch_size, frames, _, height, width = x.shape
    s_height = height >> self.ds_ratio
    s_width = width >> self.ds_ratio
    # forward h structure: (batch_size, channel, height, width)
    s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(x)
    hs = []
    for i in range(frames):
      h, s = self.cell(x[:, i, :, :, :], s)
      hs.append(h)
    outputs = []
    for i in range(self.num_fb, frames - self.num_ff):
      out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
      out = self.recons(out)
      outputs.append(out.unsqueeze(dim=1))

    return torch.cat(outputs, dim=1)