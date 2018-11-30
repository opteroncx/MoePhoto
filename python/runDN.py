# -*- coding:utf-8 -*-
import numpy as np
from imageProcess import doCrop, genGetModel
from models import NetDN, SEDN
from config import config

model_dict = {
  '15' : './model/l15/model_new.pth',
  '25' : './model/l25/model_new.pth',
  '50' : './model/l50/model_new.pth',
  'lite5' : './model/dn_lite5/model_new.pth',
  'lite10' : './model/dn_lite10/model_new.pth',
  'lite15' : './model/dn_lite15/model_new.pth'
}
ramCoef = 1 / np.array([.024, .075, .042, .22])

def dn(x, opt):
  print("doing denoise")
  return doCrop(opt, getModel(opt), x)

@genGetModel
def getModel(opt):
  print('loading', opt.model)
  if 'dn_lite' in opt.model:
    return NetDN()
  else:
    return SEDN()

##################################

def getOpt(model):
  def opt():pass
  opt.model = model_dict[model]

  conf = config.getConfig()
  modelType = 0 if model[:4] == 'lite' else 1
  cropsize = conf[modelType + 1]
  if not cropsize:
    runType, free_ram = config.getFreeMem()
    cropsize = int(np.sqrt(free_ram * ramCoef[runType * 2 + modelType]))

  if cropsize > 2048:
    cropsize = 2048
  opt.cropsize = cropsize
  print('当前denoise切块大小：', cropsize)
  opt.modelCached = getModel(opt)
  return opt
