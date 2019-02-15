from torch import load
from torchvision.transforms import Normalize
import numpy as np
from models import AODnet
from imageProcess import genGetModel, initModel
modelPath = './model/dehaze/AOD_net_epoch_relu_10.pth'
getModel = genGetModel()

def getOpt(*_):
  def opt():pass
  opt.model = modelPath
  opt.modelDef = AODnet
  dict1 = load(modelPath, map_location='cpu')
  opt.modelCached = initModel(getModel(opt), dict1)
  return opt

transform = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

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
  imgIn = transform(extractAlpha(t)(img)).unsqueeze(0)

  prediction = net(imgIn)
  dhim = prediction.squeeze()
  return mergeAlpha(t)(dhim)