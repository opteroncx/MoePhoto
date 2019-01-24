# pylint: disable=E1101
import time
from functools import reduce
import itertools
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dehaze import Dehaze
from config import config
from defaultConfig import defaultConfig
from progress import Node, updateNode

deviceCPU = torch.device('cpu')
outDir = defaultConfig['outDir'][0]
genNameByTime = lambda: '{}/output_{}.png'.format(outDir, int(time.time()))

"""unused
def check_rbga(im):
  '''convert rbga2rbg'''
  if im.shape[2] == 4:
    rbg = im[:, :, 0:3]
  else:
    rbg = im
  return rbg

def resize_image_by_pil(image, scale, resampling_method="bicubic"):
  width, height = image.shape[1], image.shape[0]
  new_width = int(width * scale)
  new_height = int(height * scale)
  if resampling_method == "bicubic":
  method = Image.BICUBIC
  elif resampling_method == "bilinear":
  method = Image.BILINEAR
  elif resampling_method == "nearest":
  method = Image.NEAREST
  else:
  method = Image.LANCZOS

  if len(image.shape) == 3 and image.shape[2] == 3:
  image = Image.fromarray(image, "RGB")
  image = image.resize([new_width, new_height], resample=method)
  image = np.asarray(image)
  elif len(image.shape) == 3 and image.shape[2] == 4:
  # RGBA images
  image = Image.fromarray(image, "RGB")
  image = image.resize([new_width, new_height], resample=method)
  image = np.asarray(image)
  else:
  image = Image.fromarray(image.reshape(height, width))
  image = image.resize([new_width, new_height], resample=method)
  image = np.asarray(image)
  image = image.reshape(new_height, new_width, 1)
  return image

def convert_rgb_to_ycbcr(image, jpeg_mode=False, max_value=255):
  if len(image.shape) < 2 or image.shape[2] == 1:
  return image

  if jpeg_mode:
  xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
  ycbcr_image = image.dot(xform.T)
  ycbcr_image[:, :, [1, 2]] += max_value / 2
  else:
  xform = np.array(
    [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
     [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
  ycbcr_image = image.dot(xform.T)
  ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
  ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)
  return ycbcr_image

def convert_rgb_to_y(image, jpeg_mode=False, max_value=255.0):
  if len(image.shape) <= 2 or image.shape[2] == 1:
  return image

  if jpeg_mode:
  xform = np.array([[0.299, 0.587, 0.114]])
  y_image = image.dot(xform.T)
  else:
  xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
  y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)
  return y_image

def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=False, max_value=255.0):
  if len(y_image.shape) == 3 and y_image.shape[2] == 3:
  y_image = y_image[:, :, 0:1]
  ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
  ycbcr_image[:, :, 0] = y_image
  ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]
  return convert_ycbcr_to_rgb(ycbcr_image)

def convert_ycbcr_to_rgb(ycbcr_image):
  rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

  rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
  rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
  xform = np.array(
  [[298.082 / 256.0, 0, 408.583 / 256.0],
   [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
   [298.082 / 256.0, 516.412 / 256.0, 0]])
  rgb_image = rgb_image.dot(xform.T)
  return rgb_image
"""
padImageReflect = torch.nn.ReflectionPad2d
unpadImage = lambda padding: lambda im: im[:, padding:-padding, padding:-padding]
cropIter = lambda length, padding, size:\
  itertools.chain(range(length - padding * 2 - size, 0, -size), [] if padding >= (length - padding * 2) % size > 0 else [0])

def doCrop(opt, model, x, padding=1, sc=1):
  pad = padImageReflect(padding)
  unpad = unpadImage(sc * padding)
  c = x.shape[0]
  hOut = x.shape[1] * sc
  wOut = x.shape[2] * sc
  squeeze = 1 if (not hasattr(opt, 'C2B')) or opt.C2B else 0
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
    s = x[:, :, topS:topS + cropsize, leftS:leftS + cropsize]
    r = model(s)[-1]
    tmp = unpad(r.squeeze(squeeze))
    tmp_image[:, topT:topT + tmp.shape[1]
      , leftT:leftT + tmp.shape[2]] = tmp

  return tmp_image

def toNumPy(bitDepth):
  dtypeT = False
  if bitDepth <= 8:
    dtype = np.uint8
  elif bitDepth <= 16:
    dtype = np.uint16
    dtypeT = np.int32
  else:
    dtype = np.int32
  def f(args):
    buffer, height, width = args
    image = np.frombuffer(buffer, dtype=dtype)
    if dtypeT:
      image = image.astype(dtypeT)
    return image.reshape((height, width, 3))
  return f

def toBuffer(bitDepth):
  if bitDepth == 8:
    dtype = np.uint8
  elif bitDepth == 16:
    dtype = np.uint16
  return lambda imt: (imt[0].astype(dtype).tostring(), imt[1].astype(np.uint8))

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

def toTorch(quant, dtype, device):
  def f(image):
    image = torch.tensor(image, dtype=torch.float, device=device) / quant  # pylint: disable=E1102
    image = image.to(dtype=dtype)
    if len(image.shape) == 3:
      return image.transpose(2, 1).transpose(1, 0)
    else:
      return image.unsqueeze(0)
  return f

def writeFile(image, name, *args):
  if not name:
    name = genNameByTime()
  Image.fromarray(image).save(name, *args)
  return name

def readFile(nodes=[]):
  def f(file):
    image = Image.open(file)
    image = np.array(image)
    for n in nodes:
      n.load *= image.size
      updateNode(n)
    if len(nodes):
      updateNode(nodes[0].parent)
    if len(image.shape) == 2 or image.shape[2] == 3 or image.shape[2] == 4:
      return image
    else:
      raise RuntimeError('Unknown image format')
  return f

def extractAlpha(t):
  def f(im):
    if im.shape[2] == 4:
      t['im'] = im[:,:,3]
      return im[:,:,:3]
    else:
      return im
  return f

def mergeAlpha(t):
  def f(im):
    if len(t):
      image = np.empty((*im.shape[:2], 4), dtype=np.uint8)
      image[:,:,:3] = im
      image[:,:,3] = t['im']
    return image
  return f

def genGetModel(f):
  def getModel(opt, cache=True):
    if hasattr(opt, 'modelCached') and cache:
      return opt.modelCached

    print('loading model {}'.format(opt.model))
    model = f(opt)
    print('reloading weights')
    weights = torch.load(opt.model)
    model.load_state_dict(weights)
    model.eval().to(dtype=config.dtype(), device=config.device())
    for param in model.parameters():
      param.requires_grad_(False)
    return model

  return getModel

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
ensemble = lambda x, es, kwargs: reduce((lambda v, t: v + t[2](doCrop(x=t[1](x), **kwargs))), zip(range(es), trans, transInv), doCrop(x=x, **kwargs))

def appendFuncs(f, node, funcs, bind=True):
  funcs.append(node.bindFunc(f) if bind else f)
  return node

import runDN
import runSR

def genProcess(opt, context=None, source='file'):
  scale = opt[0] if opt[0] else 1
  mode = opt[1] if opt[1] else 'a'
  dnmodel = opt[2] if opt[2] else 'no'
  dnseq = opt[3] if opt[3] else 'before'
  bitDepthIn = opt[4] if len(opt) > 4 and opt[4] else 8
  bitDepthOut = opt[5] if len(opt) > 5 else bitDepthIn
  if scale > 1:
    SRopt = runSR.getOpt(scale, mode, config.ensembleSR)
    SRop = {'op': 'SR', 'model': mode, 'scale': scale}
  if dnmodel != 'no':
    DNopt = runDN.getOpt(dnmodel)
    DNop = {'op': 'DN', 'model': dnmodel}
  config.getFreeMem(True)
  if not bitDepthOut:
    bitDepthOut = bitDepthIn
  quant = 1 << bitDepthIn
  funcs = []
  nodes = []
  load = 1
  last = lambda im, _: im
  if source == 'file':
    s = Node({'op': source}, learn=0, name='file')
    f = readFile(nodes)
    funcs.append(context.getFile)
  elif source == 'buffer':
    s = Node({'op': source, 'bits': bitDepthIn})
    f = toNumPy(bitDepthIn)
  appendFuncs(f, s, funcs)
  dtype = config.dtype()
  node = Node({'op': 'toTorch', 'bits': bitDepthIn, 'dtype': dtype})
  nodes.append(appendFuncs(toTorch(quant, dtype, config.device()), node, funcs))
  if (dnseq == 'before') and (dnmodel != 'no'):
    nodes.append(appendFuncs(lambda im: runDN.dn(im, DNopt), Node(DNop, name='DN'), funcs))
  load *= scale * scale
  if (scale > 1):
    if source == 'buffer' and mode == 'gan':
      nodes.append(appendFuncs(BGR2RGBTorch, Node({'op': 'Channel'}), funcs))
    nodes.append(appendFuncs(lambda im: runSR.sr(im, SRopt), Node(SRop, name='SR'), funcs))
    if source == 'buffer' and mode == 'gan':
      nodes.append(appendFuncs(BGR2RGBTorch, Node({'op': 'Channel'}, load), funcs))
  if (dnseq == 'after') and (dnmodel != 'no'):
    nodes.append(appendFuncs(lambda im: runDN.dn(im, DNopt), Node(DNop, load, name='DN'), funcs))
  funcs.append(toFloat)
  output = toOutput(bitDepthOut)
  if source != 'buffer':
    nodes.append(appendFuncs(output, Node({'op': 'toOutput', 'bits': bitDepthOut}, load), funcs))
  else:
    funcs.append(lambda im: (output(im), toOutput8(im)))
  if source == 'file':
    n = Node({'op': 'write'}, load, name='write')
    nodes.append(n)
    last = n.bindFunc(writeFile)
  elif source == 'buffer':
    nodes.append(appendFuncs(toBuffer(bitDepthOut), Node({'op': 'toBuffer', 'bits': bitDepthOut}, load), funcs))
  def process(im, name=None):
    im = reduce(apply, funcs, im)
    return last(im, name)
  return process, [s] + nodes

def clean():
  torch.cuda.empty_cache()

def dehaze(context):
  t = {}
  funcs = [context.getFile, readFile(), extractAlpha(t), Dehaze, toOutput(8), mergeAlpha(t)]
  def f(size, outputName=None):
    im = reduce(apply, funcs, size)
    return writeFile(im, outputName)
  return f