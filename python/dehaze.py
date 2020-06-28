import torch
from torchvision.transforms import Normalize
import numpy as np
from models import AODnet
from sun_demoire import Net as SUNNet
from moire_obj import Net as ObjNet
from moire_screen_gan import Net as GANNet
from imageProcess import initModel, identity, doCrop, Option
normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
ramCoef = .95 / np.array([[443., 160., 152.], [503.1, 275.34, 276.], [41951.3, 16788.7, 7029.7]])
ram2 = np.array([[[85<<23, -43520., 115/128], [-18<<20, 38272., 89/128], [-141<<20, 39488., 19/1536]],
[[61<<22, 600., 1/3920], [123<<22, 2200., 1/784], [619<<20, 1496., 1/1890]]])
mode_switch = {
  'dehaze': ('./model/dehaze/AOD_net_epoch_relu_10.pth', AODnet, ramCoef[0], 1, 8, normalize),
  'sun': ('./model/demoire/sun_epoch_200.pth', SUNNet, ramCoef[1], 9, 32, identity),
  'moire_obj': ('./model/demoire/moire_obj.pth', ObjNet, ram2[0], 9, 128, identity),
  'moire_screen_gan': ('./model/demoire/moire_screen_gan.pth', GANNet, ram2[1], 17, 512, identity),
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