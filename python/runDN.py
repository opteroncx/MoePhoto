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
ramCoef = .9 / np.array([2700., 4106.9, 2400., 7405., 1253.4, 4304.2])

def dn(x, opt):
  print("doing denoise")
  return doCrop(opt, getModel(opt), x)

@genGetModel
def getModel(opt, *args):
  print('loading', opt.model)
  if 'dn_lite' in opt.model:
    return NetDN()
  else:
    return SEDN()

##################################

def getOpt(model):
  def opt():pass
  if not model in model_dict:
    return
  opt.model = model_dict[model]

  modelType = 0 if model[:4] == 'lite' else 1
  opt.ramCoef = ramCoef[config.getRunType() * 2 + modelType]
  opt.cropsize = config.getConfig()[modelType + 1]
  print('当前denoise切块大小：', opt.cropsize)
  opt.modelCached = getModel(opt, False)
  return opt
