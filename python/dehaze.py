from torch import load
from torchvision.transforms import Normalize
import numpy as np
from models import AODnet
from sun_demoire import Net as SUNNet
from imageProcess import initModel
normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
identity = lambda x: x
mode_switch = {
  'dehaze': ('./model/dehaze/AOD_net_epoch_relu_10.pth', AODnet, normalize),
  'sun': ('./model/demoire/sun_epoch_200.pth', SUNNet, identity),
  'mddm': ('./model/demoire/mddm.pth', SUNNet, identity),
}

def getOpt(model):
  def opt():pass
  modelPath, M, transform = mode_switch[model]
  opt.model = modelPath
  opt.modelDef = M
  opt.modelCached = initModel(opt, modelPath, model)
  opt.transform = transform
  return opt

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
    else:
      return im
  return f

def Dehaze(img, opt):
  net = opt.modelCached
  t = {}
  imgIn = opt.transform(extractAlpha(t)(img)).unsqueeze(0)

  prediction = net(imgIn)
  dhim = prediction.squeeze(0)
  return mergeAlpha(t)(dhim)