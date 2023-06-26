# pylint: disable=E1101
import time
from copy import copy
from functools import reduce
from itertools import chain
from typing import Callable, List, Union
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import PIL
PIL.PILLOW_VERSION = PIL.__version__
from torchvision.transforms.functional import to_tensor
import numpy as np
from PIL import Image
from config import config
from progress import updateNode
import logging

def getAnchors(s, ns, l, pad, af, sc):
  n = l - 2 * pad
  step = 1 if l >= af(s) else max(2, int(np.ceil(ns / n)))
  start = np.arange(step, dtype=int) * n + pad
  start[0] = 0
  end = start + l
  endSc = end * sc
  if step > 1:
    start[-1] = s - af(s - end[-2] + pad)
    end[-1] = s
    clip = int((int(end[-2]) - s) * sc)
  else:
    end[-1] = af(s)
    clip = 0
  endSc[-1] = s * sc
  # clip = [0:l, pad:l - pad, ..., end[-2] - s:l]
  return start.tolist(), end.tolist(), clip, step, endSc.astype(int).tolist()

def wrapPad2D(p):
  def f(x):
    if x.dim() > 4:
      s0 = x.shape[:-3]
      s1 = x.shape[-3:]
      x = p(x.view(-1, *s1))
      return x.view(*s0, *x.shape[-3:])
    else:
      return p(x)
  return f

def getPad(aw, w, ah, h):
  if aw > 2 * w - 1 or ah > 2 * h - 1:
    tw = max(0, min(w - 1, aw - w))
    th = max(0, min(h - 1, ah - h))
    rw = aw - tw
    rh = ah - th
    return wrapPad2D(lambda x: F.pad(F.pad(x, (0, tw, 0, th), mode='reflect'), (0, rw, 0, rh)))
  else:
    return wrapPad2D(padImageReflect((0, aw - w, 0, ah - h)))

def memoryError(ram):
  raise MemoryError('Free memory space is {} bytes, which is not enough.'.format(ram))

def solveRam(m, c, k):
  if type(k) is float or k.ndim < 1:
    return m / c * k
  elif m < k[0]:
    memoryError(m)
  else: # solve k_0+k_1*x+k_2*x^2=m, where k_2>0, k_1>=0
    v = m  / c - k[0]
    return (np.sqrt(k[1] * k[1] + 4 * k[2] * v) - k[1]) / 2 / k[2]

def prepare(shape, ram, opt, pad, sc, align=8, cropsize=0):
  *_, c, h, w = shape
  n = solveRam(ram, opt.fixChannel or c, opt.ramCoef / shape[0] if shape[0] else 1.)
  af = alignF[align]
  s = af(minSize + pad * 2)
  if n < s * s:
    log.error('{} pixels can be allocated, {} required. Shape: {}, RamCoef: {}'.format(n, s * s, shape, opt.ramCoef))
    memoryError(ram)
  ph, pw = max(1, h - pad * 3), max(1, w - pad * 3)
  ns = np.arange(s / align, int(n / (align * s)) + 1, dtype=int)
  ms = (n / (align * align) / ns).astype(int)
  ns, ms = ns * align, ms * align
  nn, mn = np.ceil(ph / (ns - 2 * pad)).clip(2), np.ceil(pw / (ms - 2 * pad)).clip(2)
  nn[ns >= h] = 1
  mn[ms >= w] = 1
  ds = nn * mn # minimize number of clips
  ind = np.argwhere(ds == ds.min()).squeeze(1)
  mina = ind[np.abs(ind - len(ds) / 2).argmin()] # pick the size with ratio of width and height closer to 1
  ah, aw, acs = af(h), af(w), af(cropsize)
  ih, iw = (min(acs, ns[mina]), min(acs, ms[mina])) if cropsize > 0 else (ns[mina], ms[mina])
  ih, iw = min(ah, ih), min(aw, iw)
  startH, endH, clipH, stepH, bH = getAnchors(h, ph, ih, pad, af, sc)
  startW, endW, clipW, stepW, wH = getAnchors(w, pw, iw, pad, af, sc)
  padSc, outh, outw = int(pad * sc), int(h * sc), int(w * sc)
  if (stepH > 1) and (stepW > 1):
    padImage = identity
    unpad = identity
  elif stepH > 1:
    padImage = getPad(aw, w, 0, 0)
    unpad = lambda im: im[..., :outw]
  elif stepW > 1:
    padImage = getPad(0, 0, ah, h)
    unpad = lambda im: im[..., :outh, :]
  else:
    padImage = getPad(aw, w, ah, h)
    unpad = lambda im: im[..., :outh, :outw]
  b = ((torch.arange(padSc, dtype=config.dtype(), device=config.device()) / padSc - .5) * 9).sigmoid().view(1, -1)
  def iterClip():
    for i in range(stepH):
      top, bottom, bsc = startH[i], endH[i], bH[i]
      topT = clipH if i == stepH - 1 else (0 if i == 0 else padSc)
      for j in range(stepW):
        left, right, rsc = startW[j], endW[j], wH[j]
        leftT = clipW if j == stepW - 1 else (0 if j == 0 else padSc)
        yield (top, bottom, left, right, topT, leftT, bsc, rsc)
  return iterClip, padImage, unpad, (*shape[:-2], outh, outw), b

def blend(r, x, lt, pad, dim, blend):
  l = r.shape[dim]
  if lt < 0:
    lt = l + lt
  if lt < 1:
    return r, x
  start = lt - pad
  ls, ll = l - start, l - lt
  _, b, c = r.split([start, pad, ll], dim) # share storage
  _, bx, _ = x.split([start, pad, ll], dim)
  b = bx + blend * (b - bx)
  return torch.cat([b, c], dim), x.narrow(dim, start, ls)

def prepareOpt(opt, shape):
  sc, pad = opt.scale, opt.padding
  padSc = int(pad * sc)
  if opt.iterClip is None or opt.count > 28 or shape[0] != opt.outShape[0]:
    try:
      freeMem = config.calcFreeMem()
    except Exception:
      raise MemoryError('Can not calculate free memory.')
    opt.count = 0
    if opt.ensemble > 0:
      opt2 = copy(opt)
      opt2.iterClip, opt2.padImage, opt2.unpad, *_ = prepare(transposeShape(shape), freeMem, opt, pad, sc, opt.align, opt.cropsize)
    opt.iterClip, opt.padImage, opt.unpad, outShape, opt.blend = prepare(shape, freeMem, opt, pad, sc, opt.align, opt.cropsize)
    if opt.outShape is None:
      opt.outShape = [1, *opt.oShape[1:-2], int(sc * shape[-2]), int(sc * shape[-1])] if opt.oShape else outShape
    opt.outShape = list(opt.outShape)
    if opt.ensemble > 0:
      opt2.blend = opt.blend
      opt2.outShape = transposeShape(opt.outShape)
      opt.transposedOpt = opt2
  else:
    opt.count += 1
  return sc, padSc

def doCrop(opt, x, *args, **_):
  sc, padSc = prepareOpt(opt, x.shape)
  bl = opt.blend
  opt.outShape[0] = x.size(0)
  x = opt.padImage(opt.unsqueeze(x))
  tmp_image = x.new_empty(opt.outShape)

  for top, bottom, left, right, topT, leftT, bsc, rsc in opt.iterClip():
    s = x[..., top:bottom, left:right]
    r = opt.squeeze(opt(s, *args))
    t = tmp_image[..., int(top * sc):bsc, int(left * sc):rsc]
    q, _ = blend(*blend(opt.unpad(r), t, topT, padSc, -2, bl.t()), leftT, padSc, -1, bl)
    *_, h, w = q.shape
    tmp_image[..., bsc - h:bsc, rsc - w:rsc] = q

  return tmp_image.detach()

def resize(opt, out, pos=0, nodes=[], h=1, w=1):
  opt['update'] = True
  if not 'method' in opt:
    opt['method'] = 'bilinear'
  def f(im):
    nonlocal h, w
    if opt['update']:
      _, h, w = im.shape
      oriLoad = h * w
      h = round(h * opt['scaleH']) if 'scaleH' in opt else opt['height']
      w = round(w * opt['scaleW']) if 'scaleW' in opt else opt['width']
      newLoad = h * w
      if len(nodes):
        nodes[pos].load = im.nelement()
        newLoad /= oriLoad
        for n in nodes[pos + 1:]:
          n.multipleLoad(newLoad)
          updateNode(n)
      if out['source']:
        opt['update'] = False
    return resizeByTorch(im, w, h, opt['method'])
  return f

def restrictSize(width, height=0, method='bilinear', h=0, w=0, flag=0):
  if not height:
    height = width
  def f(im):
    nonlocal h, w, flag
    if not h:
      _, oriHeight, oriWidth = im.shape
      flag = oriHeight <= height and oriWidth <= width
      scaleH = height / oriHeight
      scaleW = width / oriWidth
      if scaleH < scaleW:
        w = round(oriWidth * scaleH)
        h = height
      else:
        h = round(oriHeight * scaleW)
        w = width
    return im if flag else resizeByTorch(im, w, h, method)
  return f

def toNumPy(bitDepth):
  if bitDepth <= 8:
    dtype = np.uint8
  elif bitDepth <= 16:
    dtype = np.uint16
  else:
    dtype = np.int32
  def f(args):
    buffer, height, width = args
    if not buffer:
      return None
    image = np.frombuffer(buffer, dtype=dtype)
    return image.reshape((height, width, 3)).astype(np.float32)
  return f

def toBuffer(bitDepth):
  if bitDepth == 8:
    dtype = np.uint8
  elif bitDepth == 16:
    dtype = np.uint16
  return lambda im: im.astype(dtype).tostring() if not im is None else None

def toFloat(image):
  if len(image.shape) == 3:  # to shape (H, W, C)
    image = image.permute(1, 2, 0)
  else:
    image = image.squeeze(0)
  return image.to(dtype=torch.float)

def toOutput(bitDepth):
  quant = 1 << bitDepth
  if bitDepth <= 8:
    dtype = torch.uint8
  elif bitDepth <= 15:
    dtype = torch.int16
  else:
    dtype = torch.int32
  def f(image):
    image = image.detach() * quant
    image.clamp_(0, quant - 1)
    return image.to(dtype=dtype, device=deviceCPU).numpy()
  return f

def toTorch(bitDepth, dtype, device):
  if bitDepth <= 8:
    return lambda image: to_tensor(image).to(dtype=dtype, device=device)
  quant = 1 << bitDepth
  return lambda image: (to_tensor(image).to(dtype=torch.float, device=device) / quant).to(dtype=dtype)

def writeFile(image, name, context, *args):
  if not name:
    name = genNameByTime()
  elif hasattr(name, 'seek'):
    name.seek(0)
  if image.shape[2] == 1:
    image = image.squeeze(2)
  image = Image.fromarray(image)
  if context.imageMode == 'P':
    image = image.quantize(palette=context.palette)
  image.save(name, *args)
  return name

def readFile(nodes=[], context=None):
  def f(file):
    image = Image.open(file)
    context.imageMode = image.mode
    if image.mode == 'P':
      context.palette = image
      image = image.convert('RGB')
    image = np.array(image)
    if context.imageMode == 'RGBA':
      context.imageMode, image = dedupeAlpha(image)
    summary = dict(mode=context.imageMode, shape=list(image.shape[:2]))
    for n in nodes:
      n.multipleLoad(image.size)
      updateNode(n)
    if len(nodes):
      p = nodes[0].parent
      updateNode(p)
      p.callback(p, summary)
    if len(image.shape) == 2:
      return image.reshape(*image.shape, 1)
    if image.shape[2] == 3 or image.shape[2] == 4:
      return image
    else:
      raise RuntimeError('Unknown image format')
  return f

def getStateDict(path):
  if not path in weightCache:
    weightCache[path] = torch.load(path, map_location='cpu')
  return weightCache[path]

def castModel(model):
  dtype = type(model).__dict__.get('castDtype', 'float16')
  if dtype == 'float32':
    _m = model.to(config.device())
  elif dtype == 'autocast':
    _m = autocast()(model.to(config.device()))
  else:
    return model.to(dtype=config.dtype(), device=config.device())
  return lambda *args, **kwargs: _m(*args, **kwargs).to(config.dtype())

def initModel(opt, weights=None, key=None, f=lambda opt: opt.modelDef(), args=[]):
  if key and key in modelCache:
    return castModel(modelCache[key])
  log.info('loading model {}'.format(opt.model))
  model = f(opt, *args)
  if weights:
    log.info('reloading weights')
    if type(weights) == str:
      weights = getStateDict(weights)
    model.load_state_dict(weights)
  for param in model.parameters():
    param.requires_grad_(False)
  model.eval()
  if key:
    modelCache[key] = model
  return castModel(model)

def getPadBy32(img, _):
  *_, oriHeight, oriWidth = img.shape
  width = ceilBy32(oriWidth)
  height = ceilBy32(oriHeight)
  pad = padImageReflect((0, width - oriWidth, 0, height - oriHeight))
  unpad = lambda im: im[:, :oriHeight, :oriWidth]
  return width, height, pad, unpad

def transposeShape(shape):
  tShape = list(shape)
  tShape[-1] = shape[-2]
  tShape[-2] = shape[-1]
  return tShape

def extractAlpha(t):
  def f(im):
    if im.shape[0] == 4:
      t['im'] = im[3]
      return im[:3]
    else:
      return im
  return f

def mergeAlpha(t):
  def f(im):
    if len(t):
      image = torch.empty((4, *im.shape[1:]), dtype=im.dtype, device=im.device)
      image[:3] = im
      image[3] = t['im']
      return image
    else:
      return im
  return f

def _RGBFilter(opt, img):
  t = {}
  imgIn = opt.prepare(extractAlpha(t)(img))

  prediction = doCrop(opt, imgIn)
  out = strengthOp(prediction, imgIn, opt.strength)
  return mergeAlpha(t)(out)
RGBFilter = lambda opt: lambda img: _RGBFilter(opt, img)

class Option():
  def __init__(self, path=''):
    self.ramCoef, self.count = 1e-3, 0
    self.padding, self.cropsize, self.align, self.fixChannel = 1, 0, 8, 1
    self.scale, self.ensemble, self.strength = 1, 0, 1.0
    self.model = path
    self.outShape, self.oShape = None, None
    self.iterClip = None
    self.prepare = identity
    self.squeeze = lambda x: x.squeeze(0)
    self.unsqueeze = lambda x: x.unsqueeze(0)

  def __call__(self, x, *args, **kwargs):
    out = self.modelCached(x, *args, **kwargs)
    if type(out) == list:
      out = out[-1]
    return out

offload = lambda b: [t.cpu() if isinstance(t, torch.Tensor) else t for t in b] if type(b) == list else b.cpu()
load2device = lambda b, device: b.to(device) if isinstance(b, torch.Tensor) else (b if b is None else [load2device(t, device) for t in b])

def DefaultStreamSource(last=None):
  flag = True
  while flag:
    t = yield
    last = t[0] if type(t) == tuple else t
    flag &= not last

class StreamState():
  def __init__(self, window=None, device=config.device(), offload=True, store=True, tensor=True, name=None, batchFunc=None, reserve=0, **_):
    self.source = DefaultStreamSource()
    next(self.source)
    self.wm1 = window - 1 if window else 0
    self.device = device
    self.tensor = tensor
    self.offload = offload and store
    self.store = store
    self.batchFunc = batchFunc if batchFunc else torch.stack if tensor else identity
    self.name = name
    self.start = 0
    self.end = 0
    self.state = []
    self.reserve = reserve # ensure enough items to pad
    self.stateR = []

  def getSize(self, size=None):
    ls = len(self.state)
    if ls < self.wm1 + (size or 1) or self.start:
      return 0
    lb = ls - self.wm1
    return min(size, lb) if size else lb

  def popBatch(self, size=1):
    r = self.getSize(size)
    if not r:
      return None
    batch = [self.batchFunc(self.state[i:i + self.wm1 + 1]) for i in range(r)] if self.wm1 else self.state[:r]
    if self.reserve:
      self.stateR = (self.stateR + self.state[r - self.reserve: r])[-self.reserve:]
    self.state = self.state[r:]
    batch = self.batchFunc(batch)
    return load2device(batch, self.device) if self.offload else batch

  def setPadding(self, padding):
    if padding > 0: self.start = padding
    elif padding < 0: self.end = padding
    return self

  def pad(self, padding: int):
    if padding == 0:
      return 0
    absPad = abs(padding)
    size = 1 + absPad * 2
    if len(self.stateR) + len(self.state) < size:
      return 0
    offset = padding - 2 if padding < 0 else 0
    ids = (torch.arange(absPad, 0, -1) + padding + offset).tolist()
    state = self.stateR + self.state
    batch = [state[i] for i in ids]
    self.state = (self.state + batch) if padding < 0 else (batch + self.state)
    return padding

  def put(self, batch: Union[torch.Tensor, List[torch.Tensor]]):
    if batch is None:
      return None
    if self.offload:
      batch = offload(batch)
    self.store and self.state.extend(t for t in batch)
    if self.start:
      self.start -= self.pad(self.start)
    return batch

  def bind(self, stateIter):
    self.source = stateIter
    return self.source

  def __len__(self):
    return self.getSize()

  def __str__(self):
    return 'StreamState {}'.format(self.name) if self.name else 'anonymous StreamState'

  def pull(self, last=None, size=None):
    if size and self.getSize() >= size:
      return 1
    t = (last, size) if size else last
    flag = not last
    try:
      self.source.send(t)
      flag = 1
    except StopIteration: pass
    if not flag and self.end:
      self.end -= self.pad(self.end)
    return flag or self.getSize() >= size

  @staticmethod
  def run(f: Callable, states, size: int, args=[], last=None, pipe=False):
    t = yield
    flag, trial = False, 2
    while True:
      last, size = t if type(t) == tuple else (t, size)
      r = min(s.getSize() for s in states)
      if r >= size or (r and flag):
        trial = 2
        r = min(r, size)
        nargs = list(args) + [s.popBatch(r) for s in states]
        out = f(*nargs, last=flag)
        t = yield out
      else:
        if flag or not pipe:
          break
        pr = [s.pull(last, size) for s in states]
        flag = not all(pr) # some source will not append
        if trial == 0:
          t = yield # don't wait if we can advance
          trial = 2
        trial -= 1

  @classmethod
  def pipe(cls, f: Callable, states, targets, size=1, args=[]):
    it = cls.run(f, states, size, args, pipe=True)
    next(it)
    itPipe = pipeFunc(it, targets, size)
    next(itPipe)
    for t in targets:
      t.bind(itPipe)
    return itPipe

def pipeFunc(it, targets, size, last=False):
  t = yield
  while True:
    flag, size = t if type(t) == tuple else (t, size)
    last |= bool(flag)
    try:
      out = it.send((last, size))
      for t in targets:
        t.put(out)
      t = yield out
    except StopIteration: break

deviceCPU = torch.device('cpu')
outDir = config.outDir
previewFormat = config.videoPreview
previewPath = config.outDir + '/.preview.{}'.format(previewFormat if previewFormat else '')
log = logging.getLogger('Moe')
modelCache = {}
weightCache = {}
fCleanCache = lambda x: torch.cuda.empty_cache() or x
genNameByTime = lambda: '{}/output_{}.png'.format(outDir, int(time.time()))
padImageReflect = torch.nn.ReflectionPad2d
identity = lambda x, *_, **__: x
ceilBy = lambda d: lambda x: (-int(x) & -d ^ -1) + 1 # d needed to be a power of 2
ceilBy32 = ceilBy(32)
minSize = 28
alignF = { 1: identity }
alignF.update((1 << k, ceilBy(1 << k)) for k in (3, 4, 5, 6, 7, 9))
resizeByTorch = lambda x, width, height, mode='bilinear':\
  F.interpolate(x.unsqueeze(0), size=(height, width), mode=mode, align_corners=False).squeeze()
clean = lambda: torch.cuda.empty_cache()
BGR2RGB = lambda im: np.stack([im[:, :, 2], im[:, :, 1], im[:, :, 0]], axis=2)
BGR2RGBTorch = lambda im: torch.stack([im[2], im[1], im[0]])
toOutput8 = toOutput(8)
dedupeAlpha = lambda x: ('RGB', x[:, :, :3]) if (255 - x[:, :, 3]).astype(dtype=np.float32).sum() < 1 else ('RGBA', x)
strengthOp = lambda x, inp, s=1: x if s == 1 else s * x + (1 - s) * inp
apply = lambda v, f: f(v)
transpose = lambda x: x.transpose(-1, -2)
flip = lambda x: x.flip(-1)
flip2 = lambda x: x.flip(-1, -2)
combine = lambda *fs: lambda x: reduce(apply, fs, x)
getTransposedOpt = lambda opt: opt.transposedOpt
trans = [transpose, flip, flip2, combine(flip, transpose), combine(transpose, flip), combine(transpose, flip, transpose), combine(flip2, transpose)]
transInv = [transpose, flip, flip2, trans[4], trans[3], trans[5], trans[6]]
which = [getTransposedOpt, identity, identity, getTransposedOpt, getTransposedOpt, identity, getTransposedOpt]
ensemble = lambda opt: lambda x: reduce((lambda v, t: (v + t[2](doCrop(t[3](opt), t[1](x)))).detach()), zip(range(opt.ensemble), trans, transInv, which), doCrop(opt, x))
split = lambda *ps: lambda x: tuple(split(*ps[1:])(c) for c in x.split(ps[0], x.ndim - len(ps))) if len(ps) else x
flat = lambda x: tuple(chain(*(flat(t) for t in x))) if len(x) and type(x[0]) is tuple else x
extend = lambda out, res, off=False: None if res is None else out.extend(tuple(offload(res) if off else res))