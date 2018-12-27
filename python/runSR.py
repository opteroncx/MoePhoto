# -*- coding:utf-8 -*-
from functools import reduce
import numpy as np
from imageProcess import ensemble, genGetModel
from models import Net2x, Net3x, Net4x
import gan
from config import config

models = (Net2x, Net3x, Net4x)
mode_switch = {
  'a2': './model/a2/model_new.pth',
  'a3': './model/a3/model_new.pth',
  'a4': './model/a4/model_new.pth',
  'p2': './model/p2/model_new.pth',
  'p3': './model/p3/model_new.pth',
  'p4': './model/p4/model_new.pth',
  'p-4': './model/gan/gan_x4.pth',
}
ramCoef = .9 / np.array([10888.4, 4971.7, 2473., 24248., 8253.9, 4698.7, 41951.3, 16788.7, 7029.7])

def data_trans(im,num):
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image, -1)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    return tranform

def data_trans_inv(im,num):
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image,-1)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    return tranform

def selfEnsemble(x,opt):
  im_list = []
  for i in range(8):
      tmp = data_trans(x,i)
      seim1=sr(tmp,opt)
      seim2=data_trans_inv(seim1,i)
      im_list.append(seim2)
  for i in range(len(im_list)):
      if i == 0:
          sum = im_list[0]
      else:
          sum += im_list[i]
  ensemble = sum/len(im_list)
  return ensemble

def sr(x, opt):
  sc = opt.scale
  sum = ensemble(x, opt.ensemble, {
    'opt': opt,
    'model': getModel(opt),
    'padding': 2 if sc == 3 else 1,
    'sc': sc
  })
  if opt.ensemble:
    return sum / (opt.ensemble + 1)
  else:
    return sum

getModel = genGetModel(lambda opt, *args: models[opt.scale - 2]())

##################################

def getOpt(scale, mode, ensemble):
  def opt():pass
  nmode = mode+str(scale)
  if not nmode in mode_switch:
    return
  opt.model = mode_switch[nmode]
  opt.scale = np.abs(scale)
  opt.ensemble = ensemble

  modelType = (np.abs(scale) - 2) * 3
  opt.ramCoef = ramCoef[config.getRunType() + modelType]
  opt.cropsize = config.getConfig()[0]
  if opt.cropsize:
    print('当前SR切块大小：', opt.cropsize)
  opt.modelCached = getModel(opt, False)
  return opt