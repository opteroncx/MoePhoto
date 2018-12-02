# -*- coding:utf-8 -*-
import numpy as np
from imageProcess import doCrop, genGetModel
from models import Net2x, Net3x, Net4x
from config import config

models = (Net2x, Net3x, Net4x)
mode_switch = {
  'a2': './model/a2/model_new.pth',
  'a3': './model/a3/model_new.pth',
  'a4': './model/a4/model_new.pth',
  'p2': './model/p2/model_new.pth',
  'p3': './model/p3/model_new.pth',
  'p4': './model/p4/model_new.pth',
}
ramCoef = 1 / np.array([.015, .05, .06, .12, .12, .24])

def sr(x, opt):
  print("doing super resolution")
  sc = opt.scale
  padding = 2 if sc == 3 else 1
  return doCrop(opt, getModel(opt), x, padding, sc)

@genGetModel
def getModel(opt):
  print('loading net {}x'.format(opt.scale))
  return models[opt.scale - 2]()

##################################

def getOpt(scale, mode):
  def opt():pass
  nmode = mode+str(scale)
  if not nmode in mode_switch:
    return
  opt.model = mode_switch[nmode]
  opt.scale = scale

  conf = config.getConfig()
  cropsize = conf[0]
  modelType = (scale - 2) * 2
  if not cropsize:
    runType, free_ram = config.getFreeMem()
    cropsize = int(np.sqrt(free_ram * ramCoef[runType + modelType]))

  if cropsize > 2048:
    cropsize = 2048
  opt.cropsize = cropsize
  print('当前SR切块大小：',cropsize)
  opt.modelCached = getModel(opt)
  return opt
