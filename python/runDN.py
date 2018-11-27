# -*- coding:utf-8 -*-
import argparse
import torch
import os
import cv2
import numpy as np
from functools import partial
import pickle
import readgpu
import psutil
#from imageProcess import convert_rgb_to_ycbcr, convert_y_and_cbcr_to_rgb
from imageProcess import doCrop, genGetModel
from models import NetDN,SEDN
from config import Config

parser = argparse.ArgumentParser(description="SEDN")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/l15/model.pth", type=str, help="model path")

cuda = torch.cuda.is_available()

def dn(x, opt):
  print("doing denoise")
  return doCrop(opt, getModel(opt), x)

@genGetModel
def getModel(opt):
  print(opt.model[:15])
  if opt.model[:15] == './model/dn_lite':
    return NetDN()
  else:
    return SEDN()

##################################

def getOpt(model):
  opt = parser.parse_args()
  model_dict = {
    '15' : './model/l15/model_new.pth',
    '25' : './model/l25/model_new.pth',
    '50' : './model/l50/model_new.pth',
    'lite5' : './model/dn_lite5/model_new.pth',
    'lite10' : './model/dn_lite10/model_new.pth',
    'lite15' : './model/dn_lite15/model_new.pth'
  }
  if not(model in model_dict):
    return {}
  opt.model = model_dict[model]

  if cuda:
    torch.cuda.empty_cache()

  conf = Config().getConfig()
  modelType = 0 if model[:4] == 'lite' else 1
  cropsize = conf[modelType + 1]
  ramCoef = 1 / np.array([.024, .075, .042, .22])
  if not cropsize:
    if cuda:
      runType = 0
      free_ram = readgpu.getGPU()
    else:
      runType = 2
      mem = psutil.virtual_memory()
      free_ram = mem.free
      free_ram = free_ram/1024**2
      # 预留内存防止系统卡死
      free_ram -= 300
      # torch.set_num_threads(1)
    cropsize = int(np.sqrt(free_ram * ramCoef[runType + modelType]))

  if cropsize > 2048:
    cropsize = 2048
  opt.cropsize = cropsize
  print('当前denoise切块大小：', cropsize)
  opt.modelCached = None
  opt.modelCached = getModel(opt)
  return opt

def main():
  print('请在flask内运行')

def dodn(im, model):
  opt = getOpt(model)
  return dn(im, opt)

if __name__=="__main__":
  main()
