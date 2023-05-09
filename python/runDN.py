# -*- coding:utf-8 -*-
import numpy as np
from models import NetDN, SEDN
from MPRNet import MPRNet
from NAFNet import NAFNet
from imageProcess import initModel, Option, extractAlpha, mergeAlpha, doCrop
from config import config

ramCoef = .95 / np.array([[2700., 2400., 1253.4], [4106.9, 7405., 4304.2], [60493., 8400., 4288.], [3409., 693., 457.], [6815., 1169., 692.]])
mode_switch = {
  '15': ('./model/l15/model_new.pth', SEDN, ramCoef[1], 1, 7, 8),
  '25': ('./model/l25/model_new.pth', SEDN, ramCoef[1], 1, 7, 8),
  '50': ('./model/l50/model_new.pth', SEDN, ramCoef[1], 1, 7, 8),
  'lite5': ('./model/dn_lite5/model_new.pth', NetDN, ramCoef[0], 1, 7, 8),
  'lite10': ('./model/dn_lite10/model_new.pth', NetDN, ramCoef[0], 1, 7, 8),
  'lite15': ('./model/dn_lite15/model_new.pth', NetDN, ramCoef[0], 1, 7, 8),
  'MPRNet_denoising': ('./model/MPRNet/model_denoising.pth', (lambda: MPRNet(n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32)), ramCoef[2], 0, 7, 8),
  'NAFNet_32': ('./model/NAFNet/NAFNet-SIDD-width32.pth', (lambda: NAFNet(width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])), ramCoef[3], 0, 15, 16),
  'NAFNet_64': ('./model/NAFNet/NAFNet-SIDD-width64.pth', (lambda: NAFNet(width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])), ramCoef[4], 0, 15, 16)
}

##################################

def getOpt(optDN):
  model = optDN['model']
  if not model in mode_switch:
    return None
  opt = Option(mode_switch[model][0])
  _, opt.modelDef, ramCoef, sd, opt.padding, opt.align = mode_switch[model]

  opt.ramCoef = ramCoef[config.getRunType()]
  opt.cropsize = config.getConfig()[1 if model[:4] == 'lite' else 2]
  opt.modelCached = initModel(opt, opt.model, 'DN' + model)
  if sd:
    opt.fixChannel = 0
    opt.squeeze = lambda x: x.squeeze(sd)
    opt.unsqueeze = lambda x: x.unsqueeze(sd)
  opt.padding = 15
  return opt

def denoise(opt, img):
  t = {}
  imgIn = extractAlpha(t)(img)

  prediction = doCrop(opt, imgIn)
  return mergeAlpha(t)(prediction)