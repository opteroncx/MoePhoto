import torch
from torchvision.transforms import Normalize
import numpy as np
from models import AODnet
from sun_demoire import Net as SUNNet
from imageProcess import initModel, getPadBy32, identity
_normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
normalize = lambda x: _normalize(x.squeeze(0)).unsqueeze(0)
ramCoef = .95 / np.array([[443., 160., 152.], [503.1, 275.34, 276.], [41951.3, 16788.7, 7029.7]])
mode_switch = {
  'dehaze': ('./model/dehaze/AOD_net_epoch_relu_10.pth', AODnet, ramCoef[0], lambda *_: (normalize, identity)),
  'sun': ('./model/demoire/sun_epoch_200.pth', SUNNet, ramCoef[1], getPadBy32),
  'mddm': ('./model/demoire/mddm.pth', SUNNet, ramCoef[2], getPadBy32),
}

def getOpt(model):
  def opt():pass
  modelPath, opt.modelDef, opt.ram, opt.prepare = mode_switch[model]
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
      image = torch.empty((4, *im.shape[1:]), dtype=im.dtype)
      image[:3] = im
      image[3] = t['im']
      return image
    else:
      return im
  return f

def Dehaze(img, opt):
  net = opt.modelCached
  t = {}
  *_, transform, revert = opt.prepare(img, opt)
  imgIn = transform(extractAlpha(t)(img).unsqueeze(0))

  prediction = net(imgIn)
  out = revert(prediction.squeeze(0))
  return mergeAlpha(t)(out)