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
from defaultConfig import defaultConfig
from progress import Node, updateNode
import logging

deviceCPU = torch.device('cpu')
outDir = defaultConfig['outDir'][0]
log = logging.getLogger('Moe')
genNameByTime = lambda: '{}/output_{}.png'.format(outDir, int(time.time()))
padImageReflect = torch.nn.ReflectionPad2d
unpadImage = lambda padding: lambda im: im[:, padding:-padding, padding:-padding]
cropIter = lambda length, padding, size:\
  itertools.chain(range(length - padding * 2 - size, 0, -size), [] if padding >= (length - padding * 2) % size > 0 else [0])

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

  return tmp_image

PILResamples = {
  'nearest': Image.NEAREST,
  'bilinear': Image.BILINEAR,
  'bicubic': Image.BICUBIC
}
def resizeByPIL(x, width, height, mode='bicubic'):
  if x.shape[0] == 1:
    x = x.squeeze(0)
  y = Image.fromarray(toOutput8(toFloat(x))).resize((width, height), resample=PILResamples[mode])
  y = np.array(y)
  if len(y.shape) == 2:
    y = y.reshape(*y.shape, 1)
  return to_tensor(y).to(dtype=x.dtype, device=x.device)

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
    if type(inp) != type(None):
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
  return lambda im: im.astype(dtype).tostring() if type(im) != type(None) else None

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
    image = image * quant
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

def genGetModel(f=lambda opt: opt.modelDef()):
  def getModel(opt):
    log.info('loading model {}'.format(opt.model))
    return f(opt)
  return getModel

def initModel(model, weights=None):
  if weights:
    log.info('reloading weights')
    model.load_state_dict(weights)
  for param in model.parameters():
    param.requires_grad_(False)
  return model.eval().to(dtype=config.dtype(), device=config.device())

identity = lambda x, *_: x
clean = lambda: torch.cuda.empty_cache()
NonNullWrap = lambda f: lambda x: f(x) if type(x) != type(None) else None
BGR2RGB = lambda im: np.stack([im[:, :, 2], im[:, :, 1], im[:, :, 0]], axis=2)
BGR2RGBTorch = lambda im: torch.stack([im[2], im[1], im[0]])
toOutput8 = toOutput(8)
apply = lambda v, f: f(v)
applyNonNull = lambda v, f: f(v) if type(v) != type(None) else None
transpose = lambda x: x.transpose(-1, -2)
flip = lambda x: x.flip(-1)
flip2 = lambda x: x.flip(-1, -2)
combine = lambda *fs: lambda x: reduce(apply, fs, x)
trans = [transpose, flip, flip2, combine(flip, transpose), combine(transpose, flip), combine(transpose, flip, transpose), combine(flip2, transpose)]
transInv = [transpose, flip, flip2, trans[4], trans[3], trans[5], trans[6]]
ensemble = lambda x, es, kwargs: reduce((lambda v, t: v + t[2](doCrop(x=t[1](x), **kwargs))), zip(range(es), trans, transInv), doCrop(x=x, **kwargs))
previewPath = defaultConfig['outDir'][0] + '/.preview.png'

def toInt(o, keys):
  for key in keys:
    if key in o:
      o[key] = int(o[key])

def appendFuncs(f, node, funcs, wrap=True):
  g = node.bindFunc(f)
  funcs.append(NonNullWrap(g) if wrap else g)
  return node

import runDN
import runSR
import runSlomo
import dehaze
from worker import context

fPreview = [
  0,
  toFloat,
  toOutput8,
  (lambda im: im.astype(np.uint8)),
  0,
  lambda im: writeFile(im, context.shared, 'PNG'),
  lambda *_: context.root.trace(0, preview=previewPath, fileSize=context.shared.tell())]
funcPreview = lambda im: reduce(applyNonNull, fPreview, im)

def procInput(source, bitDepth, fs, out):
  out['load'] = 1
  node = Node({'op': 'toTorch', 'bits': bitDepth})
  fs.append(NonNullWrap(node.bindFunc(toTorch(bitDepth, config.dtype(), config.device()))))
  return fs, [node], out

def procDN(opt, out, *_):
  DNopt = opt['opt']
  node = Node(dict(op='DN', model=opt['model']), out['load'])
  if 'name' in opt:
    node.name = opt['name']
  return [NonNullWrap(node.bindFunc(lambda im: runDN.dn(im, DNopt)))], [node], out

def convertChannel(out):
  out['channel'] = 0
  fs=[]
  return fs, [appendFuncs(BGR2RGBTorch, Node(dict(op='Channel')), fs)]

def procSR(opt, out, *_):
  load = out['load']
  scale = opt['scale']
  mode = opt['model']
  SRopt = opt['opt']
  if not scale > 1:
    raise TypeError('Invalid scale setting for SR.')
  out['load'] = load * scale * scale
  fs, ns = convertChannel(out) if out['channel'] and mode == 'gan' else ([], [])
  ns.append(appendFuncs(lambda im: runSR.sr(im, SRopt), Node(dict(op='SR', model=mode), load), fs))
  if 'name' in opt:
    ns[-1].name = opt['name']
  return fs, ns, out

def procSlomo(opt, out, *_):
  load = out['load']
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  node = Node(dict(op='slomo'), load, opt['sf'], name=opt['name'] if 'name' in opt else None)
  return fs + [runSlomo.doSlomo], ns + [node], out

def procDehaze(opt, out, *_):
  load = out['load']
  dehazeOpt = opt['opt']
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  node = Node(dict(op='dehaze'), load, name=opt['name'] if 'name' in opt else None)
  ns.append(appendFuncs(lambda im: dehaze.Dehaze(im, dehazeOpt), node, fs))
  return fs, ns, out

def procResize(opt, out, nodes):
  node = Node(dict(op='resize', mode=opt['method']), 1, name=opt['name'] if 'name' in opt else None)
  return [node.bindFunc(resize(opt, out, len(nodes), nodes))], [node], out

def procOutput(opt, out, *_):
  load = out['load']
  node0 = Node(dict(op='toFloat'), load)
  bitDepthOut = out['bitDepth']
  node1 = Node(dict(op='toOutput', bits=bitDepthOut), load, name=opt['name'] if 'name' in opt else None)
  fOutput = node1.bindFunc(toOutput(bitDepthOut))
  fs = [NonNullWrap(node0.bindFunc(toFloat)), NonNullWrap(fOutput)]
  ns = [node0, node1]
  if out['source']:
    fPreview[0] = restrictSize(2048)
    fs1 = [node0.bindFunc(toFloat), fOutput]
    def o(im):
      res = reduce(applyNonNull, fs1, im)
      funcPreview(im)
      return [res]
    fs = [o]
    if out['channel']:
      fPreview[4] = BGR2RGB
    else:
      fPreview[4] = identity
      ns.append(appendFuncs(BGR2RGB, Node(dict(op='Channel')), fs1, False))
      out['channel'] = 1
    ns.append(appendFuncs(toBuffer(bitDepthOut), Node(dict(op='toBuffer', bits=bitDepthOut), load), fs1, False))
  return fs, ns, out

procs = dict(
  file=(lambda _, _0, nodes:
    procInput('file', 8, [context.getFile, readFile(nodes)], dict(bitDepth=8, channel=0, source=0))),
  buffer=(lambda opt, *_:
    procInput('buffer', opt['bitDepth'], [toNumPy(opt['bitDepth'])], dict(bitDepth=opt['bitDepth'], channel=1, source=1))),
  DN=procDN, SR=procSR, output=procOutput, slomo=procSlomo, dehaze=procDehaze, resize=procResize
  )

def genProcess(steps, root=True, outType=None):
  funcs=[]
  nodes=[]
  last = identity
  rf = lambda im: reduce(apply, funcs, im)
  if root:
    for opt in filter((lambda opt: opt['op'] == 'SR'), steps):
      toInt(opt, ['scale'])
      opt['opt'] = runSR.getOpt(opt['scale'], opt['model'], config.ensembleSR)
    for opt in filter((lambda opt: opt['op'] == 'resize'), steps):
      toInt(opt, ['scaleW', 'scaleH', 'width', 'height'])
    for opt in filter((lambda opt: opt['op'] == 'DN'), steps):
      opt['opt'] = runDN.getOpt(opt['model'])
    for opt in filter((lambda opt: opt['op'] == 'dehaze'), steps):
      opt['opt'] = dehaze.getOpt()
    slomos = [*filter((lambda opt: opt['op'] == 'slomo'), steps)]
    for opt in slomos:
      toInt(opt, ['sf'])
      opt['opt'] = runSlomo.getOpt(opt)
    if len(slomos):
      slomos[-1]['opt'].notLast = 0
    if steps[-1]['op'] != 'output':
      steps.append(dict(op='output'))
    config.getFreeMem(True)
    process = lambda im, name=None: last(rf(im), name)
  else:
    process = rf
  for i, opt in enumerate(steps):
    op = opt['op']
    fs, ns, outType = procs[op](opt, outType, nodes)
    funcs.extend(fs)
    nodes.extend(ns)
    if op == 'slomo':
      if i + 1 < len(steps):
        f, nodesAfter = genProcess(steps[i + 1:], False, outType)
      else:
        f = identity
        nodesAfter = []
      slomoOpt = opt['opt']
      slomo = funcs[-1](f, nodes[-1])
      funcs[-1] = windowWrap(lambda data: slomo(data, slomoOpt), slomoOpt, 2)
      nodeAfter = Node({}, total=opt['sf'], learn=0)
      for node in nodesAfter:
        nodeAfter.append(node)
      nodes.append(nodeAfter)
      break
  if root and steps[0]['op'] == 'file':
    n = Node({'op': 'write'}, outType['load'])
    nodes.append(n)
    last = n.bindFunc(writeFile)
  return process, nodes