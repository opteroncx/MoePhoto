# -*- coding:utf-8 -*-
import numpy as np
from imageProcess import initModel, Option
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

##################################

def getOpt(optDN):
  model = optDN['model']
  if not model in mode_switch:
    return
  opt = Option(mode_switch[model][0])
  opt.modelDef = mode_switch[model][1]

  opt.ramCoef = mode_switch[model][2][config.getRunType()]
  opt.cropsize = config.getConfig()[1 if model[:4] == 'lite' else 2]
  opt.modelCached = initModel(opt, opt.model, 'DN' + model)
  opt.squeeze = lambda x: x.squeeze(1)
  opt.unsqueeze = lambda x: x.unsqueeze(1)
  opt.padding = 15
  return opt