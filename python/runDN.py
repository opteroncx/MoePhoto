# -*- coding:utf-8 -*-
import numpy as np
from torch import load
from imageProcess import ensemble, genGetModel, initModel
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

dn = lambda x, opt: ensemble(x, 0, { 'opt': opt, 'model': opt.modelCached })

getModel = genGetModel(lambda opt, *args: NetDN() if 'dn_lite' in opt.model else SEDN())

##################################

def getOpt(model):
  def opt():pass
  if not model in model_dict:
    return
  opt.model = model_dict[model]

  modelType = 0 if model[:4] == 'lite' else 1
  opt.ramCoef = ramCoef[config.getRunType() * 2 + modelType]
  opt.cropsize = config.getConfig()[modelType + 1]
  if opt.cropsize:
    print('当前denoise切块大小：', opt.cropsize)
  opt.modelCached = initModel(getModel(opt), load(opt.model))
  return opt
