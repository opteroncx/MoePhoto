# pylint: disable=E1101
import time
from functools import reduce
import itertools
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import numpy as np
from PIL import Image
from config import config
from progress import updateNode
import logging

def doCrop(opt, model, x, padding=1, sc=1):
  pad = padImageReflect(padding)
  unpad = unpadImage(sc * padding)
  squeeze = 1 if (not hasattr(opt, 'C2B')) or opt.C2B else 0
  c = x.shape[0]
  baseFlag = hasattr(opt, 'mode') and opt.mode == 'lite.old'
  if baseFlag:
    c = c >> 1
    base = padImageReflect(sc * padding)(x[c:].unsqueeze(squeeze))
    x = x[:c,:x.shape[1] >> 1,:x.shape[2] >> 1]
  hOut = x.shape[1] * sc
  wOut = x.shape[2] * sc
  x = pad(x.unsqueeze(squeeze))
  _, _, h, w = x.shape
  tmp_image = torch.zeros([c, hOut, wOut]).to(x)

  cropsize = opt.cropsize
  if not cropsize:
    try:
      freeRam = config.calcFreeMem()
      cropsize = int(np.sqrt(freeRam * opt.ramCoef / c))
    except:
      raise MemoryError()
  if cropsize > 2048:
    cropsize = 2048
  if not cropsize > 32:
    raise MemoryError()
  size = cropsize - 2 * padding

  for topS, leftS in itertools.product(cropIter(h, padding, size), cropIter(w, padding, size)):
    leftT = leftS * sc
    topT = topS * sc
    bottomS = topS + cropsize
    rightS = leftS + cropsize
    s = x[:, :, topS:bottomS, leftS:rightS]
    r = model(s, base[:, :, topT:bottomS * sc,leftT:rightS * sc]) if baseFlag else model(s)[-1]
    tmp = unpad(r.squeeze(squeeze))
    tmp_image[:, topT:topT + tmp.shape[1]
      , leftT:leftT + tmp.shape[2]] = tmp

  return tmp_image.detach()

resizeByTorch = lambda x, width, height, mode='bilinear':\
  F.interpolate(x.unsqueeze(0), size=(height, width), mode=mode, align_corners=False).squeeze()

def resize(opt, out, pos=0, nodes=[]):
  opt['update'] = True
  if not 'method' in opt:
    opt['method'] = 'bilinear'
  h = w = 1
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

def restrictSize(width, height=0, method='bilinear'):
  if not height:
    height = width
  h = w = flag = 0
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

def windowWrap(f, opt, window=2):
  cache = []
  maxBatch = 1 << 7
  h = 0
  getData = lambda: [cache[i:i + window] for i in range(h - window + 1)]
  def init(r=False):
    nonlocal h, cache
    if r and window > 1:
      cache = cache[h - window + 1:h] + [0 for _ in range(maxBatch)]
      h = window - 1
    else:
      cache = [0 for _ in range(window + maxBatch - 1)]
      h = 0
  init()
  def g(inp=None):
    nonlocal h
    b = min(max(1, opt.batchSize), maxBatch)
    if not inp is None:
      cache[h] = inp
      h += 1
      if h >= window + b - 1:
        data = getData()
        init(True)
        return f(data)
    elif h >= window:
      data = getData()
      init()
      return f(data)
  return g

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
      return
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
    image = image.transpose(0, 1).transpose(1, 2)
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

def writeFile(image, name, *args):
  if not name:
    name = genNameByTime()
  elif hasattr(name, 'seek'):
    name.seek(0)
  if image.shape[2] == 1:
    image = image.squeeze(2)
  Image.fromarray(image).save(name, *args)
  return name

def readFile(nodes=[]):
  def f(file):
    image = Image.open(file)
    image = np.array(image)
    for n in nodes[1:]:
      n.multipleLoad(image.size)
      updateNode(n)
    if len(nodes):
      updateNode(nodes[0].parent)
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

def initModel(opt, weights=None, key=None, f=lambda opt: opt.modelDef()):
  if key and key in modelCache:
    return modelCache[key].to(dtype=config.dtype(), device=config.device())
  log.info('loading model {}'.format(opt.model))
  model = f(opt)
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
  return model.to(dtype=config.dtype(), device=config.device())

def toInt(o, keys):
  for key in keys:
    if key in o:
      o[key] = int(o[key])

def getPadBy32(img, _):
  *_, oriHeight, oriWidth = img.shape
  width = ceilBy32(oriWidth)
  height = ceilBy32(oriHeight)
  pad = padImageReflect((0, width - oriWidth, 0, height - oriHeight))
  unpad = lambda im: im[:, :oriHeight, :oriWidth]
  return width, height, pad, unpad

deviceCPU = torch.device('cpu')
outDir = config.outDir
previewFormat = config.videoPreview
previewPath = config.outDir + '/.preview.{}'.format(previewFormat if previewFormat else '')
log = logging.getLogger('Moe')
modelCache = {}
weightCache = {}
genNameByTime = lambda: '{}/output_{}.png'.format(outDir, int(time.time()))
padImageReflect = torch.nn.ReflectionPad2d
unpadImage = lambda padding: lambda im: im[:, padding:-padding, padding:-padding]
cropIter = lambda length, padding, size:\
  itertools.chain(range(length - padding * 2 - size, 0, -size), [] if padding >= (length - padding * 2) % size > 0 else [0])
identity = lambda x, *_: x
clean = lambda: torch.cuda.empty_cache()
BGR2RGB = lambda im: np.stack([im[:, :, 2], im[:, :, 1], im[:, :, 0]], axis=2)
BGR2RGBTorch = lambda im: torch.stack([im[2], im[1], im[0]])
toOutput8 = toOutput(8)
apply = lambda v, f: f(v)
transpose = lambda x: x.transpose(-1, -2)
flip = lambda x: x.flip(-1)
flip2 = lambda x: x.flip(-1, -2)
combine = lambda *fs: lambda x: reduce(apply, fs, x)
trans = [transpose, flip, flip2, combine(flip, transpose), combine(transpose, flip), combine(transpose, flip, transpose), combine(flip2, transpose)]
transInv = [transpose, flip, flip2, trans[4], trans[3], trans[5], trans[6]]
ensemble = lambda x, es, kwargs: reduce((lambda v, t: v + t[2](doCrop(x=t[1](x), **kwargs))), zip(range(es), trans, transInv), doCrop(x=x, **kwargs)).detach()
ceilBy = lambda d: lambda x: (-int(x) & -d ^ -1) + 1 # d needed to be a power of 2
ceilBy32 = ceilBy(32)