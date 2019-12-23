import torch
from torchvision.transforms import Normalize
import numpy as np
from models import AODnet
from sun_demoire import Net as SUNNet
from imageProcess import initModel, identity, doCrop, Option
normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
ramCoef = .95 / np.array([[443., 160., 152.], [503.1, 275.34, 276.], [41951.3, 16788.7, 7029.7]])
mode_switch = {
  'dehaze': ('./model/dehaze/AOD_net_epoch_relu_10.pth', AODnet, ramCoef[0], 1, 8, normalize),
  'sun': ('./model/demoire/sun_epoch_200.pth', SUNNet, ramCoef[1], 62, 32, identity),
  'mddm': ('./model/demoire/mddm.pth', SUNNet, ramCoef[2], 31, 32, identity),
}

def getOpt(optDe):
  model = optDe.get('model', 'dehaze')
  opt = Option()
  modelPath, opt.modelDef, opt.ram, opt.padding, opt.align, opt.prepare = mode_switch[model]
  opt.model = modelPath
  opt.modelCached = initModel(opt, modelPath, model)
  return opt

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

def Dehaze(opt, img):
  t = {}
  imgIn = opt.prepare(extractAlpha(t)(img))

  prediction = doCrop(opt, imgIn)
  return mergeAlpha(t)(prediction)