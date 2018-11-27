# -*- coding:utf-8 -*-
# pylint: disable=E1101
import argparse
import torch
import numpy as np
from functools import partial
import readgpu
import psutil
#from imageProcess import resize_image_by_pil, convert_rgb_to_ycbcr, convert_rgb_to_y, convert_y_and_cbcr_to_rgb
from imageProcess import doCrop, genGetModel
from models import Net2x, Net3x, Net4x
from config import Config

parser = argparse.ArgumentParser(description="MoePhoto")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/sr24/model.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--net", default='null', type=str, help="network file")

cuda = torch.cuda.is_available()

def sr(x, opt):
  print("doing super resolution")
  sc = opt.scale
  padding = 2 if sc == 3 else 1
  # tmp = resize_image_by_pil(x, sc)
  # gt_yuv = convert_rgb_to_ycbcr(tmp)
  # x = convert_rgb_to_y(x)
  return doCrop(opt, getModel(opt), x, padding, sc)
  # convert_y_and_cbcr_to_rgb(tmp_image.numpy(), gt_yuv[:, :, 1:3])

@genGetModel
def getModel(opt):
  if opt.scale == 2:
    print('loading net 2x')
    return Net2x()
  if opt.scale == 3:
    print('loading net 3x')
    return Net3x()
  if opt.scale == 4:
    print('loading net 4x')
    return Net4x()

##################################

def getOpt(scale, mode):
  opt = parser.parse_args()
  mode_switch = {
    'a2': './model/a2/model_new.pth',
    'a3': './model/a3/model_new.pth',
    'a4': './model/a4/model_new.pth',
    'p2': './model/p2/model_new.pth',
    'p3': './model/p3/model_new.pth',
    'p4': './model/p4/model_new.pth',
  }
  nmode = mode+str(scale)
  if not (nmode in mode_switch):
    return {}
  opt.model = mode_switch[nmode]
  opt.scale = scale
  if cuda:
    torch.cuda.empty_cache()

  conf = Config().getConfig()
  cropsize = conf[0]
  modelType = (scale - 2) * 2
  ramCoef = 1 / np.array([.015, .05, .06, .12, .12, .24])
  if not cropsize:
    if cuda:
      runType = 0
      free_ram = readgpu.getGPU()
    else:
      runType = 1
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
  print('当前SR切块大小：',cropsize)
  opt.modelCached = None
  opt.modelCached = getModel(opt)
  return opt

def dosr(im, scale, mode):
  opt = getOpt(scale, mode)
  return sr(im, opt)
def main():
  print('请在flask内运行')

if __name__=="__main__":
  main()