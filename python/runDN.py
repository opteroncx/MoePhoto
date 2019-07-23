# -*- coding:utf-8 -*-
import numpy as np
from torch import load
from imageProcess import ensemble, initModel
from models import NetDN, SEDN
from config import config

ramCoef = .9 / np.array([[2700., 2400., 1253.4], [4106.9, 7405., 4304.2]])
mode_switch = {
  '15': ('./model/l15/model_new.pth', SEDN, ramCoef[1]),
  '25': ('./model/l25/model_new.pth', SEDN, ramCoef[1]),
  '50': ('./model/l50/model_new.pth', SEDN, ramCoef[1]),
  'lite5': ('./model/dn_lite5/model_new.pth', NetDN, ramCoef[0]),
  'lite10': ('./model/dn_lite10/model_new.pth', NetDN, ramCoef[0]),
  'lite15': ('./model/dn_lite15/model_new.pth', NetDN, ramCoef[0]),
}

dn = lambda x, opt: ensemble(x, 0, { 'opt': opt, 'model': opt.modelCached })


##################################

def getOpt(model):
  def opt():pass
  if not model in mode_switch:
    return
  opt.model = mode_switch[model][0]
  opt.modelDef = mode_switch[model][1]

  opt.ramCoef = mode_switch[model][2][config.getRunType()]
  opt.cropsize = config.getConfig()[1 if model[:4] == 'lite' else 2]
  opt.modelCached = initModel(opt, opt.model, 'DN' + model)
  return opt