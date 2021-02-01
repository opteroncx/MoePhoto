import sys
sys.path.append('./python')
sys.path.append('./site-packages/nvidia-ml-py')
import os
from os.path import split, splitext
import torch
from PIL import Image
import numpy as np
from imageProcess import toTorch, readFile, initModel, toFloat, toOutput, ensemble, writeFile, Option
from config import config
from time import perf_counter

from moire_obj import Net
from dehaze import getOpt
modelName = 'moire_obj'
test = False
inputFolder = '../test-pics'
refFile = 0 #'test/1566005911.7879605_ci.png'

def context():pass
# opt = Option(('test/{}.pth' if test else 'model/demoire/{}.pth').format(modelName))
# opt.padding = 31
# opt.ramCoef = 1 / 8000.
# opt.align = 128
opt = getOpt(dict(model=modelName))
opt.modelCached = initModel(opt, weights=opt.model, f=lambda _: Net())
toTorch = lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).to(dtype=config.dtype(), device=config.device()) / 256
time = 0.0
for pic in os.listdir(inputFolder):
  original = toTorch(readFile(context=context)(inputFolder + '/' + pic))
  ref = toTorch(readFile(context=context)(refFile + '/' + pic)) if refFile else original
  start = perf_counter()
  y = ensemble(opt)(original)
  time += perf_counter() - start
  print(pic, float(y.mean(dtype=torch.float)), float((y - ref).abs().mean(dtype=torch.float)))
  out = toOutput(8)(toFloat(y))
  writeFile(out, 'download/{}.{}.png'.format(splitext(split(pic)[1])[0], modelName), context)
print(time)