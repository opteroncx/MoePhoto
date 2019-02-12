from functools import reduce
import numpy as np
from torch import load
from imageProcess import ensemble, genGetModel, initModel, interpolate
from models import Net2x, Net3x, Net4x
from gan import RRDB_Net
from MoeNet_lite import Net
from config import config

mode_switch = {
  'a2': ('./model/a2/model_new.pth', Net2x),
  'a3': ('./model/a3/model_new.pth', Net3x),
  'a4': ('./model/a4/model_new.pth', Net4x),
  'p2': ('./model/p2/model_new.pth', Net2x),
  'p3': ('./model/p3/model_new.pth', Net3x),
  'p4': ('./model/p4/model_new.pth', Net4x),
  'gan4': ('./model/gan/gan_x4.pth', lambda: RRDB_Net(upscale=4)),
  'lite2': ('./model/lite/lite_new.pth', Net)
}
ramCoef = .9 / np.array([10888.4, 4971.7, 2473., 24248., 8253.9, 4698.7, 41951.3, 16788.7, 7029.7, 15282.4, 5496., 4186.6, 8035., 2496.4, 1346.])

def sr(x, opt):
  sc = opt.scale
  if opt.mode == 'lite':
    x = interpolate(x)
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

getModel = genGetModel(lambda opt, *args: opt.modelDef())

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

  modelType = (scale - 2) * 3
  if mode[:3] == 'gan':
    modelType = 9
  elif mode == 'lite':
    modelType = 12
  opt.ramCoef = ramCoef[config.getRunType() + modelType]
  opt.cropsize = config.getConfig()[0]
  if opt.cropsize:
    print('当前SR切块大小：', opt.cropsize)
  opt.modelCached = getModel(opt)
  initModel(opt.modelCached, load(opt.model))
  return opt