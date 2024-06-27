from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from progress import Node
from imageProcess import ceilBy, StreamState, identity, initModel, trans, transInv, doCrop, prepareOpt
from runSlomo import newOpt, getOptS, getOptP, makeStreamFunc
from config import config

Channels = dict(
  S=[24, 36, 54, 72],
  M=[32, 48, 72, 96],
  L=[(64, 7), 96, 144, 192]
)
SideChannels = dict(S=24, M=32, L=64)
RefTime = 2

class Warp(nn.Module):
  def __init__(self, padding_mode='zeros'):
    super(Warp, self).__init__()
    self.padding_mode = padding_mode

  def setSize(self, H, W):
    kwarg = {'indexing': 'ij'} if torch.__version__ >= '1.10' else {}
    gridY, gridX = torch.meshgrid(torch.linspace(-1.0, 1.0, H), torch.linspace(-1.0, 1.0, W), **kwarg)
    self.kh = 2.0 / (H - 1)
    self.kw = 2.0 / (W - 1)
    self.register_buffer('grid', torch.cat((gridX, gridY), 0).view(1, 2, H, W))
    return self

  def forward(self, img, flow):
    flow_ = torch.stack([flow[:, 0, :, :] * self.kw, flow[:, 1, :, :] * self.kh], 1)
    grid = (self.grid + flow_).permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, padding_mode=self.padding_mode, mode='bilinear', align_corners=True)

resize = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

convrelu = lambda in_channels, out_channels, kernel_size=3, stride=1: nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size >> 1, bias=True),
    nn.PReLU(out_channels)
  )

Pyramid = lambda in_channels, out_channels, kernel_size=3: nn.Sequential(
    convrelu(in_channels, out_channels, kernel_size, 2),
    convrelu(out_channels, out_channels, 3, 1)
  )

class IFRNetEncoder(nn.Module):
  def __init__(self, _, chs, ramCoef):
    super(IFRNetEncoder, self).__init__()
    chsOut = [cOut if type(cOut) is tuple else (cOut,) for cOut in chs]
    chsIn = [3] + [c[0] for c in chsOut]
    self.pyramids = nn.ModuleList([Pyramid(cIn, *cOut) for cIn, cOut in zip(chsIn, chsOut)])
    self.encoders = [newOpt(f, ramCoef, align=8, padding=8, scale=.5, oShape=(1, cOut[0], .5, .5))
                     for f, cOut in zip(self.pyramids, chsOut)]

  def forward(self, inp, **_):
    r = reduce(lambda r, f: [doCrop(f, r[0])] + r, self.encoders, [inp])[:-1] # small to large
    return [[r[j][i] for j in range(4)] for i in range(len(inp))]

class ResBlock(nn.Module):
  def __init__(self, in_channels, side_channels, bias=True):
    super(ResBlock, self).__init__()
    self.side_channels = side_channels
    self.conv1 = convrelu(in_channels, in_channels)
    self.conv2 = convrelu(side_channels, side_channels)
    self.conv3 = convrelu(in_channels, in_channels)
    self.conv4 = convrelu(side_channels, side_channels)
    self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    self.prelu = nn.PReLU(in_channels)

  def forward(self, x):
    out = self.conv1(x)
    out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :])
    out = self.conv3(out)
    out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :])
    out = self.prelu(x + self.conv5(out))
    return out

Decoder = lambda in_channels, out_channels, side_channels, add_channels=4: nn.Sequential(
    convrelu(in_channels + add_channels, in_channels),
    ResBlock(in_channels, side_channels),
    nn.ConvTranspose2d(in_channels, 4 + out_channels, 4, 2, 1, bias=True)
  )

FlowDecoder = lambda _, in_channels, side_channels, add_channels=4: nn.Sequential(
    convrelu(in_channels + add_channels, in_channels),
    ResBlock(in_channels, side_channels),
    nn.ConvTranspose2d(in_channels, 4, 4, 2, 1, bias=True)
  )

ensembling = lambda opt, fs, x: sum(f[1](y) for f, y in zip(fs, doCrop(opt, torch.cat([f[0](x) for f in fs])).chunk(len(fs))))
class IFRNetDecoder(nn.Module):
  def __init__(self, _, chs, side_channels=24, ensemble=0, ramCoef=None):
    super(IFRNetDecoder, self).__init__()
    chsD = [k[0] if type(k) is tuple else k for k in reversed(chs)]
    chsOut = chsD[1:] + [4]
    chsIn = [k * 3 if i else k * 2 for i, k in enumerate(chsD)]
    chsAdd = [1] + [4] * 3
    self.decoders = nn.ModuleList([Decoder(cIn, cOut, side_channels, cAdd) for cIn, cOut, cAdd in zip(chsIn, chsOut, chsAdd)])
    self.decode = [newOpt(f, ramCoef, align=8, padding=7, scale=2, oShape=(1, 4 + cOut, 2, 2))
                   for f, cOut in zip(self.decoders, chsOut)]
    self.warps = [Warp('border') for _ in range(4)]
    assert ensemble < 8
    self.ensemble = ensemble
    self.ramCoef = ramCoef
    self.flows = []
    self.chsIn = chsIn
    self.chsAdd = chsAdd
    self.chsSide = side_channels
    if ensemble:
      self.fs, self.fTs = [[(trans[i], transInv[i]) for i in t if i < ensemble] for t in ((1, 2, 5), (0, 3, 4, 6))]

  def setSize(self, h, w, x):
    for i in range(4):
      self.warps[i].setSize(h >> (3 - i), w >> (3 - i)).to(x)
    if self.ensemble:
      for i, d, cIn, cAdd in zip(range(4), self.decoders, self.chsIn, self.chsAdd):
        sd = d.state_dict()
        sd['2.weight'] = sd['2.weight'][:, :4]
        sd['2.bias'] = sd['2.bias'][:4]
        opt = newOpt(0, self.ramCoef, align=8, padding=7, scale=2, oShape=(1, 4, 2, 2))
        flow = initModel(opt, sd, 0, FlowDecoder, [cIn, self.chsSide, cAdd])
        opt.modelCached = flow
        opt.ensemble = self.ensemble
        prepareOpt(opt, [1, 4, h >> (4 - i), w >> (4 - i)])
        self.flows.append(opt)
    return self

  def forward(self, x, embt, **_):
    embt = [t[0] for t in embt]
    ids = sum(([i] * len(t) for i, t in enumerate(embt)), [])
    *_, c, h, w = x[0].shape
    args = (x[0][ids].view(-1, 2 * c, h, w), torch.cat(embt).view(-1, 1, 1, 1).repeat(1, 1, h, w))
    for i in range(4):
      if i:
        up_flow0, up_flow1, ft_ = args
        ft = x[i][ids]
        warp = self.warps[i - 1]
        f0_warp = warp(ft[:, 0], up_flow0)
        f1_warp = warp(ft[:, 1], up_flow1)
        args = (ft_, f0_warp, f1_warp, up_flow0, up_flow1)
      xF = torch.cat(args, 1)
      out = doCrop(self.decode[i], xF)
      if self.ensemble:
        opt0, opt1 = self.flows[i], self.flows[i].transposedOpt
        out[:, :4] = (out[:, :4] + ensembling(opt1, self.fTs, xF) +
        (ensembling(opt0, self.fs, xF) if len(self.fs) else 0)) / (self.ensemble + 1)
      up_flow0_ = out[:, :2]
      up_flow1_ = out[:, 2:4]
      ft_ = out[:, 4:]
      if i: # inplace + to change `out`
        up_flow0_ += 2.0 * resize(up_flow0)
        up_flow1_ += 2.0 * resize(up_flow1)
      if i == 3: break
      args = (up_flow0_, up_flow1_, ft_)
    # break `out` into Tuple[Tensor] to keep pairing with input data
    return list(out.split([len(t) for t in embt]))

calcMean = lambda inp, **_: inp.mean((1, 2, 3), keepdim=True)
normializeInp = lambda inp, mean_, **_: inp - mean_

def postOut(warp, inp, inpN, mean_, embt, out, **_):
  outLens = [len(t[0]) for t in embt]
  ids = sum(([i] * k for i, k in enumerate(outLens)), [])
  inpR = inpN[ids]
  e = torch.cat([t[0] for t in embt]).view(-1, 1, 1, 1)
  mean_p = mean_[ids]
  mean_p = (1 - e) * mean_p[:, 0] + e * mean_p[:, 1]
  assert inpR.shape[0] == out.shape[0] == mean_p.shape[0]
  up_flow0, up_flow1, up_mask, up_res_1 = out[:, :2], out[:, 2:4], out[:, 4:5], out[:, 5:]
  up_mask_1 = torch.sigmoid(up_mask)

  img0, img1 = inpR[:, 0], inpR[:, 1]
  img0_warp = warp(img0, up_flow0)
  img1_warp = warp(img1, up_flow1)
  imgt_merge = up_mask_1 * (img0_warp - img1_warp) + img1_warp + mean_p
  imgt_pred = imgt_merge + up_res_1
  preds = imgt_pred.clamp(0, 1).split(outLens)
  res = [inp[:1, 0]] * embt[0][1] if embt[0][1] else []
  for img, p, t in zip(inp, preds, embt):
    res.append(p)
    img1 = img[1:]
    for _ in range(t[2]):
      res.append(img1)
  return torch.cat(res)

batchFeatures = lambda x: [torch.stack([s[i] for s in x]) for i in range(4)]

hardshrink = lambda k, c: 0 if abs(k - c) < 1e-6 else k
getEmbWeight = lambda i, c, dtype, device: torch.arange(-hardshrink(i % c, c), 1 + 1e-6, c, dtype=dtype, device=device)[1:]
getEmbStruct = lambda t: (t[:-1], 0, 1) if float(t[-1]) + 1e-6 > 1 else (t, 0, 0)
class EmbtState():
  def __init__(self, sf):
    assert sf >= 1
    self.c = 1 / sf
    self.count = 0
    self.dtype = torch.float
    self.device = torch.device('cpu')

  def to(self, x: torch.Tensor, count: int):
    self.dtype = x.dtype
    self.device = x.device
    self.count = count

  def getSize(self, size=1 << 30):
    return size

  def pull(self, last=None, *_, **__):
    return not last

  """
  Returns List[Tuple[3]|None]: [(
    embbeding tensor of the 2 input frames,
    whether or how many to keep the first input frame,
    whether or how many to keep the last input frame
  )]
  """
  def popBatch(self, size=1):
    res = [getEmbStruct(getEmbWeight(i, self.c, self.dtype, self.device)) for i in range(self.count, self.count + size)]
    if not self.count:
      res[0] = (res[0][0], 1, res[0][2])
    self.count += size
    return res

class Deduper(nn.Module):
  NullOutput = [(None,) * 5]
  def __init__(self, low, high):
    super(Deduper, self).__init__()
    self.feature = None
    self.embt = None
    self.state = [[None]]
    self.high = high
    self.low = low
    self.skips = 0

  def concat(self, embt):
    self.skips += 1
    s1 = self.state[1]
    self.state[1] = (torch.concat((s1[0], torch.ones(s1[2]).to(s1[0]) * self.skips, embt[0] + self.skips)), s1[1] + embt[1], embt[2])

  def forward(self, *args, last=None):
    assert len(args[1]) == 1
    newState = [a[0] for a in args]
    x = newState[0]
    embt = newState[1]
    x1 = self.state[0][0]
    if x1 is None:
      self.state = newState
      return self.NullOutput
    sim = F.cosine_similarity(x1.view(-1), x[0].view(-1), dim=0)
    if sim > self.high: # skip the last input frame
      self.concat(embt)
      if not last:
        return self.NullOutput
    s = self.state
    if sim < self.low: # repeat the first input frame
      e0 = s[1]
      e1 = (torch.empty(0).to(e0[0]), e0[1] + len(e0[0]), e0[2])
      s[1] = e1
    if self.skips:
      s[1] = (s[1][0] / (self.skips + 1), s[1][1], s[1][2])
    self.state = newState
    self.skips = 0
    return [s, newState] if last else [s]

extract = lambda n: lambda inp, **_: [item[n] for item in inp if item[n] is not None]

encoderRamCoef = dict(
  S=.4 / np.array([297.5, 259.21, 147.57]),
  M=.4 / np.array([300., 321.31, 183.78]),
  L=.4 / np.array([600., 759.29, 505.91])
)
decoderRamCoef = dict(
  S=.1 / np.array([730.1, 838.46, 432.02]),
  M=.1 / np.array([948.61, 826.99, 580.82]),
  L=.1 / np.array([1822.63, 1600.88, 1131.67])
)
modelPaths = dict(
  S='./model/IFRNet/IFRNet_S_GoPro.pth',
  M='./model/IFRNet/IFRNet_GoPro.pth',
  L='./model/IFRNet/IFRNet_L_GoPro.pth',
)
modules = {
  'encoder': dict(weight='encoder', streams=['features'], f=IFRNetEncoder),
  'decoder': dict(weight='decoder', streams=['decoded'], f=IFRNetDecoder)
}
def getOpt(option):
  model = option['model'][-1]
  chs = Channels[model]
  ensemble = option.get('ensemble', 0)
  modules['encoder']['args'] = (chs, encoderRamCoef[model][config.getRunType()])
  modules['encoder']['ramCoef'] = encoderRamCoef[model]
  modules['decoder']['args'] = (chs, SideChannels[model], ensemble, decoderRamCoef[model][config.getRunType()])
  modules['decoder']['ramCoef'] = decoderRamCoef[model]
  opt = getOptP(getOptS(modelPaths[model], modules, {}))
  opt.sf = option['sf']
  opt.dedupe = option.get('dedupe', False)
  opt.dedupeLow = option.get('low', .5)
  opt.dedupeHigh = option.get('high', .993)
  print(option)
  return opt

def initFunc(opt, x):
  *_, h, w = x.shape
  width = ceilBy(16)(w)
  height = ceilBy(16)(h)
  opt.pad = nn.ReflectionPad2d((0, width - w, 0, height - h))
  opt.unpad = lambda im: im[:, :h, :w]
  opt.decoder.setSize(height, width, x)
  opt.embt.to(x, opt.start)
  opt.end = 0
  return height, width

def doSlomo(func, node, opt):
  load = opt.sf - 1
  nodes = [Node({'IFRNet': 'encode'}), Node({'IFRNet': 'decode'}, load=load), Node({'IFRNet': 'post'}, load=load)]
  inp = StreamState(offload=False)
  inps = [*(StreamState(offload=False) for _ in range(3 if opt.dedupe else 2)), StreamState(2, offload=False)]
  StreamState.pipe(identity, [inp], inps[:3])
  means = [*(StreamState(offload=False) for _ in range(2 if opt.dedupe else 1)), StreamState(2, offload=False)]
  StreamState.pipe(calcMean, [inps[0]], means[:2])
  inpNs = [*(StreamState(offload=False) for _ in range(2 if opt.dedupe else 1)), StreamState(2, offload=False)]
  StreamState.pipe(normializeInp, [inps[1], means[0]], inpNs[:2])
  features = StreamState(2, tensor=False, offload=False, batchFunc=batchFeatures)
  opt.embt = EmbtState(opt.sf)
  if opt.dedupe:
    ft1 = StreamState(tensor=False, offload=False)
    ims = [StreamState(tensor=False, offload=False) for _ in range(5)]
    deduper = Deduper(opt.dedupeLow, opt.dedupeHigh)
    StreamState.pipe(deduper, [ft1, opt.embt, inps[2], inpNs[1], means[1]], ims)
    emb1 = StreamState(tensor=False, offload=False)
    for i, state in enumerate((features, emb1, inps[-1], inpNs[-1], means[-1])):
      StreamState.pipe(extract(i), ims[i:i + 1], [state])
  opt.features = StreamState.pipe(nodes[0].bindFunc(opt.encoder), [inpNs[0]], [ft1 if opt.dedupe else features])
  embs = [StreamState(tensor=False, offload=False) for _ in range(2)]
  StreamState.pipe(identity, [emb1 if opt.dedupe else opt.embt], embs)
  decode = StreamState(tensor=False, offload=False, batchFunc=torch.cat)
  opt.decoded = StreamState.pipe(nodes[1].bindFunc(opt.decoder), [features, embs[0]], [decode])
  pred = StreamState(store=False)
  opt.out = StreamState.pipe(nodes[2].bindFunc(postOut), [inps[-1], inpNs[-1], means[-1], embs[1], decode], [pred], args=[opt.decoder.warps[-1]])
  return makeStreamFunc(func, node, opt, nodes, 'slomo', [], initFunc, inp.put)