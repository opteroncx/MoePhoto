'''
super slomo
code refered from https://github.com/avinashpaliwal/Super-SloMo.git
'''
# pylint: disable=E1101
import logging
import torch
from slomo import UNet, backWarp
from imageProcess import initModel, getStateDict, getPadBy32, doCrop, identity, Option
from config import config

log = logging.getLogger('Moe')
modelPath = './model/slomo/SuperSloMo.ckpt'
RefTime = 2
WindowSize = 2
ramCoef = [.95 / x for x in (2700., 828., 2700., 822., 1338., 360.)]
getFlowComp = lambda *_: UNet(6, 4)
getFlowIntrp = lambda *_: UNet(20, 5)
getFlowBack = lambda opt: backWarp(opt.width, opt.height, config.device(), config.dtype())
getBatchSize = lambda load, ramCoef: max(1, int((config.calcFreeMem() / load) * ramCoef))
modules = dict(
  flowComp={'weight': 'state_dictFC', 'f': getFlowComp, 'outShape': (1, 4, 1, 1)},
  ArbTimeFlowIntrp={'weight': 'state_dictAT', 'f': getFlowIntrp, 'outShape': (1, 5, 1, 1)})

def newOpt(func, ramCoef, align=32, padding=45, **_):
  opt = Option()
  opt.modelCached = lambda x: (func(x),) # compatibility with doCrop calling f(x)[-1]
  opt.ramCoef = ramCoef
  opt.align = align
  opt.padding = padding
  opt.squeeze = identity
  opt.unsqueeze = identity
  return opt

def getOptS(modelPath, modules, ramCoef):
  opt = Option(modelPath)
  weights = getStateDict(modelPath)
  opt.outStart = 0
  opt.modulesCount = len(modules)
  opt.ramOffset = config.getRunType() * len(modules)
  for i, key in enumerate(modules):
    wKey = modules[key]['weight']
    constructor = modules[key].get('f', 0)
    rc = modules[key]['ramCoef'][opt.ramOffset] if 'ramCoef' in modules[key] else ramCoef[opt.ramOffset + i]
    opt.__dict__[key] =\
      newOpt(initModel(opt, weights[wKey], key, constructor), rc, **modules[key])\
      if constructor else None
  return opt

def setOutShape(modules, opt, height, width, bf=getBatchSize):
  load = width * height
  od = opt.__dict__
  for key, o in modules.items():
    batchSize = bf(load, od[key].ramCoef)
    if 'outShape' in o:
      q = o['outShape']
      od[key].outShape = [batchSize, *q[1:-2], int(height * q[-2]), int(width * q[-1])]
      if 'staticDims' in o:
        for i in o['staticDims']:
          od[key].outShape[i] = q[i]
    if 'streams' in o:
      for name in o['streams']:
        od[name].send((None, batchSize))
  return opt

def getOpt(option):
  opt = getOptS(modelPath, modules, ramCoef)
  opt.flowBackWarp = None
  opt.batchSize = 0
  opt.sf = option['sf']
  if opt.sf < 2:
    raise RuntimeError('Error: --sf/slomo factor has to be at least 2')
  return opt

def doSlomo(func, node, opt):
  # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
  # - Removed per channel mean subtraction for CPU.

  def f(data):
    node.reset()
    node.trace(0, p='slomo start')
    batchSize = len(data)
    if not batchSize or len(data[0]) < 2:
      return
    if opt.flowBackWarp is None:
      width, height, opt.pad, opt.unpad = getPadBy32(data[0][0], opt)
      opt.width = width
      opt.height = height
      opt.flowBackWarp = initModel(opt, None, None, getFlowBack)
      setOutShape(modules, opt, height, width)
      opt.batchSize = opt.flowComp.outShape[0]
      log.info('Slomo batch size={}'.format(opt.batchSize))
    else:
      width, height = opt.width, opt.height
    flowBackWarp = opt.flowBackWarp

    opt.flowComp.outShape[0] = batchSize
    opt.ArbTimeFlowIntrp.outShape[0] = batchSize
    sf = opt.sf
    tempOut = [0 for _ in range(batchSize * sf + 1)]
    # Save reference frames
    tempOut[0] = data[0][0]
    for i, frames in enumerate(data):
      tempOut[(i + 1) * sf] = frames[1]

    # Load data
    I0 = opt.pad(torch.stack([frames[0] for frames in data]))
    I1 = opt.pad(torch.stack([frames[1] for frames in data]))
    flowOut = doCrop(opt.flowComp, torch.cat((I0, I1), dim=1))
    F_0_1 = flowOut[:,:2,:,:]
    F_1_0 = flowOut[:,2:,:,:]
    node.trace()

    # Generate intermediate frames
    for intermediateIndex in range(1, sf):
      t = intermediateIndex / sf
      temp = -t * (1 - t)
      fCoeff = (temp, t * t, (1 - t) * (1 - t), temp)
      wCoeff = (1 - t, t)

      F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
      F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

      g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
      g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

      intrpOut = doCrop(opt.ArbTimeFlowIntrp, torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

      F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
      F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
      V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
      V_t_1   = 1 - V_t_0

      g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
      g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

      Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

      # Save intermediate frame
      for i in range(batchSize):
        tempOut[intermediateIndex + i * sf] = opt.unpad(Ft_p[i].detach())

      node.trace()

    if data is None and opt.outEnd:
      tempOut = tempOut[:opt.outEnd]
      opt.outEnd = 0
    for i in range(opt.outStart, len(tempOut)):
      tempOut[i] = func(tempOut[i])
    res = []
    for item in tempOut[opt.outStart:]:
      if type(item) == list:
        res.extend(item)
      elif not item is None:
        res.append(item)
    opt.outStart = max(0, opt.outStart - len(tempOut))
    return res
  return f