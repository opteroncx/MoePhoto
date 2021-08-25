import numpy as np
from imageProcess import ensemble, initModel, Option
from models import Net2x, Net3x, Net4x, RRDBNet
from MoeNet_lite2 import Net
from config import config

#(CPU:float32, GPU:float32, GPU:float16)
#[Net2x, Net3x, Net4x, RRDB_Net, MoeNet_lite2, MoeNet_lite.old, MoeNet_lite2x4, MoeNet_lite2x8]
# ramCoef = .9 / np.array([[10888.4, 4971.7, 2473.], [24248., 8253.9, 6120.], [41951.3, 16788.7, 7029.7], [15282.4, 5496., 4186.6], [3678., 4712.1, 3223.2], [8035., 2496.4, 1346.], [10803., 10944., 5880.5], [40915., 50049., 27899]])
ramCoef = .9 / np.array([[10888.4, 4971.7, 2473.], [24248., 8253.9, 6120.], [41951.3, 16788.7, 7029.7], [8e3, 4787, 4e3], [3678., 4712.1, 3223.2], [8035., 2496.4, 1346.], [10803., 10944., 5880.5], [40915., 50049., 27899], [3594, 1277, 1051]])
mode_switch = {
  'a2': ('./model/a2/model_new.pth', Net2x, ramCoef[0]),
  'a3': ('./model/a3/model_new.pth', Net3x, ramCoef[1]),
  'a4': ('./model/a4/model_new.pth', Net4x, ramCoef[2]),
  'p2': ('./model/p2/model_new.pth', Net2x, ramCoef[0]),
  'p3': ('./model/p3/model_new.pth', Net3x, ramCoef[1]),
  'p4': ('./model/p4/model_new.pth', Net4x, ramCoef[2]),
  'gan2': ('./model/gan/RealESRGAN_x2plus.pth', lambda: RRDBNet(num_in_ch=3, num_out_ch=3, scale=2), ramCoef[8]),
  'gan4': ('./model/gan/RealESRGAN_x4plus.pth', lambda: RRDBNet(num_in_ch=3, num_out_ch=3, scale=4), ramCoef[3]),
  #'lite.old2': ('./model/lite/lite.pth', NetOld, ramCoef[5]),
  'lite2': ('./model/lite/model.pth', Net, ramCoef[4]),
  'lite4': ('./model/lite/model_4.pth', lambda: Net(upscale=4), ramCoef[6]),
  'lite8': ('./model/lite/model_8.pth', lambda: Net(upscale=8), ramCoef[7])
}

sr = lambda opt: (lambda x: ensemble(opt)(x) / (opt.ensemble + 1)) if opt.ensemble else ensemble(opt)

##################################

def getOpt(optSR):
  opt = Option()
  opt.mode = optSR['model']
  opt.scale = optSR['scale']
  nmode = opt.mode+str(opt.scale)
  if not nmode in mode_switch:
    return
  if opt.mode[:3] != 'gan':
    opt.squeeze = lambda x: x.squeeze(1)
    opt.unsqueeze = lambda x: x.unsqueeze(1)
  opt.padding = 9 if opt.scale == 3 else 5
  opt.model = mode_switch[nmode][0]
  opt.modelDef = mode_switch[nmode][1]
  opt.ensemble = optSR['ensemble'] if 'ensemble' in optSR and (0 <= optSR['ensemble'] <= 7) else config.ensembleSR

  opt.ramCoef = mode_switch[nmode][2][config.getRunType()]
  opt.cropsize = config.getConfig()[0]
  opt.modelCached = initModel(opt, opt.model, 'SR' + nmode)
  return opt