import torch
from torchvision.transforms import Normalize
import numpy as np
from models import AODnet
from sun_demoire import Net as SUNNet
from moire_obj import Net as ObjNet
from moire_screen_gan import Net as GANNet
from MPRNet import MPRNet
from imageProcess import initModel, identity, doCrop, Option
from config import config
normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
ramCoef = .95 / np.array([[1329., 480., 456.], [1509.3, 826.02, 828.], [69981, 9616, 5040], [30069, 3960, 2120]])
ram2 = np.array([[[18196, 32868., 25/32], [-89<<20, 6336., 533 / 768], [-68<<22, 7264., 282/829]],
[[-98<<16, 6640., 1/771], [68<<19, 1152., 0], [53<<18, 1088., 0]]])
mode_switch = {
  'dehaze': ('./model/dehaze/AOD_net_epoch_relu_10.pth', AODnet, ramCoef[0], 1, 8, normalize),
  'sun': ('./model/demoire/sun_epoch_200.pth', SUNNet, ramCoef[1], 9, 32, identity),
  'moire_obj': ('./model/demoire/moire_obj.pth', ObjNet, ram2[0], 9, 128, identity),
  'moire_screen_gan': ('./model/demoire/moire_screen_gan.pth', GANNet, ram2[1], 17, 512, identity),
  'MPRNet_deblurring': ('./model/MPRNet/model_deblurring.pth', MPRNet, ramCoef[2], 7, 8, identity),
  'MPRNet_deraining': ('./model/MPRNet/model_deraining.pth', (lambda: MPRNet(n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16)), ramCoef[3], 7, 8, identity),
}

def getOpt(optDe):
  model = optDe.get('model', 'dehaze')
  opt = Option()
  modelPath, opt.modelDef, ramCoef, opt.padding, opt.align, opt.prepare = mode_switch[model]
  opt.ramCoef = ramCoef[config.getRunType()]
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