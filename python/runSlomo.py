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
ramCoef = [.9 / x for x in (450., 138., 450., 137., 223., 60.)]
getFlowComp = lambda *_: UNet(6, 4)
getFlowIntrp = lambda *_: UNet(20, 5)
getFlowBack = lambda opt: backWarp(opt.width, opt.height, config.device(), config.dtype())

def newOpt(f, ramType):
  opt = Option()
  opt.modelCached = lambda x: (f(x),)
  opt.ramCoef = ramCoef[config.getRunType() * 2 + ramType]
  opt.align = 32
  opt.padding = 45
  opt.squeeze = identity
  opt.unsqueeze = identity
  return opt

def getOpt(option):
  opt = Option(modelPath)
  # Initialize model
  dict1 = getStateDict(modelPath)
  flowComp = initModel(opt, dict1['state_dictFC'], 'flowComp', getFlowComp)
  ArbTimeFlowIntrp = initModel(opt, dict1['state_dictAT'], 'ArbTimeFlowIntrp', getFlowIntrp)
  opt.sf = option['sf']
  opt.firstTime = 1
  opt.notLast = 1
  opt.batchSize = 0
  opt.flowBackWarp = None
  opt.optFlow = newOpt(flowComp, 0)
  opt.optArb = newOpt(ArbTimeFlowIntrp, 1)
  if opt.sf < 2:
    raise RuntimeError('Error: --sf/slomo factor has to be at least 2')
  return opt

def getBatchSize(option):
  return max(1, int((config.calcFreeMem() / option['load'] / 6) * ramCoef[config.getRunType() * 2]))

def doSlomo(func, node, opt):
  # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
  # - Removed per channel mean subtraction for CPU.

  def f(data):
    node.reset()
    node.trace(0, p='slomo start')
    if opt.flowBackWarp is None:
      width, height, opt.pad, opt.unpad = getPadBy32(data[0][0], opt)
      opt.width = width
      opt.height = height
      opt.flowBackWarp = initModel(opt, None, None, getFlowBack)
    else:
      width, height = opt.width, opt.height
    flowBackWarp = opt.flowBackWarp

    if not opt.batchSize:
      opt.batchSize = getBatchSize({'load': width * height})
      log.info('Slomo batch size={}'.format(opt.batchSize))
    batchSize = len(data)
    opt.optFlow.outShape = (batchSize, 4, height, width)
    opt.optArb.outShape = (batchSize, 5, height, width)
    sf = opt.sf
    tempOut = [0 for _ in range(batchSize * sf + 1)]
    # Save reference frames
    if opt.notLast or opt.firstTime:
      tempOut[0] = func(data[0][0])
      outStart = 0
    else:
      outStart = 1
    for i, frames in enumerate(data):
      tempOut[(i + 1) * sf] = frames[1]

    # Load data
    I0 = opt.pad(torch.stack([frames[0] for frames in data]))
    I1 = opt.pad(torch.stack([frames[1] for frames in data]))
    flowOut = doCrop(opt.optFlow, torch.cat((I0, I1), dim=1))
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

      intrpOut = doCrop(opt.optArb, torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

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
      tempOut[intermediateIndex] = func(tempOut[intermediateIndex])

    for i in range(sf, len(tempOut)):
      tempOut[i] = func(tempOut[i])
    res = []
    for item in tempOut[outStart:]:
      if type(item) == list:
        res.extend(item)
      elif not item is None:
        res.append(item)
    opt.firstTime = 0
    return res
  return f