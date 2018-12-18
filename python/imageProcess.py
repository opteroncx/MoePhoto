# pylint: disable=E1101
import functools
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from dehaze import Dehaze
from config import config

deviceCPU = torch.device('cpu')

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

cropNum = lambda length, padding, size: ((length - padding) // size) + (1 if length % size > 2 * padding else 0)

def doCrop(opt, model, x, padding=1, sc=1):
  pad = padImageReflect(padding)
  hOut = x.shape[1] * sc
  wOut = x.shape[2] * sc
  x = pad(x.unsqueeze(1))
  c, _, h, w = x.shape
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
  print('cropsize==',cropsize)
  num_across = cropNum(h, padding, size)
  num_up = cropNum(w, padding, size)
  leftS = h - padding * 2
  for _i in range(num_across):
    leftS -= size
    if leftS < 0:
      leftS = 0
    leftT = leftS * sc
    topS = w - padding * 2
    for _j in range(num_up):
      topS -= size
      if topS < 0:
        topS = 0
      topT = topS * sc
      s = x[:, :, leftS:leftS + cropsize, topS:topS + cropsize]
      r = model(s)[-1]
      tmp = r.squeeze(
        1)[:, sc * padding:-sc * padding, sc * padding:-sc * padding]
      tmp_image[:, leftT:leftT + tmp.shape[1], topT:topT +
            tmp.shape[2]] = tmp

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
  return lambda image: image.astype(dtype).tostring()

def toOutput(bitDepth):
  quant = 1 << bitDepth
  if bitDepth <= 8:
    dtype = torch.uint8
  elif bitDepth <= 15:
    dtype = torch.int16
  else:
    dtype = torch.int32
  def f(image):
    if len(image.shape) == 3:  # to shape (H, W, C)
      image = image.transpose(0, 1).transpose(1, 2)
    else:
      image = image.squeeze(0)
    image = image.to(dtype=torch.float) * quant
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

def writeFile(image, name):
  if not os.path.exists('download'):
    os.mkdir('download')
  result = cv2.imwrite(name, image)
  if result:
    return name
  else:
    raise RuntimeError('Failed to save image')

def readFile(file):
  image = Image.open(file)
  image = np.array(image)
  if len(image.shape) == 2:
    return image
  elif image.shape[2] == 3:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  elif image.shape[2] == 4:
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
  else:
    raise RuntimeError('Unknown image format')

def extractAlpha(t):
  def f(im):
    if im.shape[2] == 4:
      t.append(im[:,:,3])
      return im[:,:,:3]
    else:
      return im
  return f

def mergeAlpha(t):
  def f(im):
    if len(t):
      im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
      im[:,:,3] = t[0]
    return im
  return f

def genGetModel(f):
  def getModel(opt, cache=True):
    if hasattr(opt, 'modelCached') and cache:
      return opt.modelCached

    model = f(opt)
    print('reloading weights')
    weights = torch.load(opt.model)
    model.load_state_dict(weights)
    model.eval().to(dtype=config.dtype(), device=config.device())
    for param in model.parameters():
      param.requires_grad_(False)
    return model

  return getModel

import runDN
import runSR
apply = lambda v, f: f(v)

def genProcess(scale=1, mode='a', dnmodel='no', dnseq='before', source='image', bitDepthIn=8, bitDepthOut=0):
  SRopt = runSR.getOpt(scale, mode)
  DNopt = runDN.getOpt(dnmodel)
  config.getFreeMem(True)
  if not bitDepthOut:
    bitDepthOut = bitDepthIn
  quant = 1 << bitDepthIn
  funcs = []
  last = lambda im, _: im
  if source == 'file':
    funcs.append(readFile)
  elif source == 'buffer':
    funcs.append(toNumPy(bitDepthIn))
  funcs.append(toTorch(quant, config.dtype(), config.device()))
  if (dnseq == 'before') and (dnmodel != 'no'):
    funcs.append(lambda im: runDN.dn(im, DNopt))
  if (scale > 1):
    funcs.append(lambda im: runSR.sr(im, SRopt))
  if (dnseq == 'after') and (dnmodel != 'no'):
    funcs.append(lambda im: runDN.dn(im, DNopt))
  funcs.append(toOutput(bitDepthOut))
  if source == 'file':
    last = writeFile
  elif source == 'buffer':
    funcs.append(toBuffer(bitDepthOut))
  def process(im, name=None):
    im = functools.reduce(apply, funcs, im)
    return last(im, name)
  return process

def clean():
  torch.cuda.empty_cache()

def dehaze(fileName, outputName):
  t = []
  im = functools.reduce(apply, [readFile, extractAlpha(t), Dehaze, toOutput(8), mergeAlpha(t)], fileName)
  return writeFile(im, outputName)
