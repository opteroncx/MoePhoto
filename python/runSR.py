from functools import reduce
import numpy as np
from torch import load
from imageProcess import ensemble, genGetModel, initModel, resize
from models import Net2x, Net3x, Net4x
from gan import RRDB_Net
from MoeNet_lite2 import Net
from config import config

#(CPU:float32, GPU:float32, GPU:float16)
#[Net2x, Net3x, Net4x, RRDB_Net, MoeNet_lite2, MoeNet_lite.old, MoeNet_lite2x4]
ramCoef = .9 / np.array([[10888.4, 4971.7, 2473.], [24248., 8253.9, 4698.7], [41951.3, 16788.7, 7029.7], [15282.4, 5496., 4186.6], [3678., 4712.1, 3223.2], [8035., 2496.4, 1346.], [10803., 10944., 5880.5]])
mode_switch = {
  'a2': ('./model/a2/model_new.pth', Net2x, ramCoef[0]),
  'a3': ('./model/a3/model_new.pth', Net3x, ramCoef[1]),
  'a4': ('./model/a4/model_new.pth', Net4x, ramCoef[2]),
  'p2': ('./model/p2/model_new.pth', Net2x, ramCoef[0]),
  'p3': ('./model/p3/model_new.pth', Net3x, ramCoef[1]),
  'p4': ('./model/p4/model_new.pth', Net4x, ramCoef[2]),
  'gan4': ('./model/gan/gan_x4.pth', lambda: RRDB_Net(upscale=4), ramCoef[3]),
  #'lite.old2': ('./model/lite/lite.pth', NetOld, ramCoef[5]),
  'lite2': ('./model/lite/model.pth', Net, ramCoef[4]),
  'lite4': ('./model/lite/model_4.pth', lambda: Net(upscale=4), ramCoef[6])
}

def sr(x, opt):
  sc = opt.scale
  if opt.mode == 'lite.old':
    x = resize(dict(scaleH=2, scaleW=2), {'source': 0})(x)
  sum = ensemble(x, opt.ensemble, {
    'opt': opt,
    'model': opt.modelCached,
    'padding': 2 if sc == 3 else 1,
    'sc': sc
  })
  if opt.ensemble:
    return sum / (opt.ensemble + 1)
  else:
    return sum

getModel = genGetModel()

##################################

def getOpt(scale, mode, ensemble):
  def opt():pass
  nmode = mode+str(scale)
  if not nmode in mode_switch:
    return
  opt.C2B = mode[:3] != 'gan'
  opt.mode = mode
  opt.model = mode_switch[nmode][0]
  opt.modelDef = mode_switch[nmode][1]
  opt.scale = scale
  opt.ensemble = ensemble

  opt.ramCoef = mode_switch[nmode][2][config.getRunType()]
  opt.cropsize = config.getConfig()[0]
  if opt.cropsize:
    print('当前SR切块大小：', opt.cropsize)
  opt.modelCached = initModel(getModel(opt), load(opt.model, map_location='cpu'))
  return opt
