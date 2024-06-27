import logging
from imageProcess import initModel, getStateDict, identity, Option, extend
from config import config

log = logging.getLogger('Moe')
getBatchSize = lambda load, ramCoef, freeMem: max(1, int((freeMem / load) * ramCoef))

def newOpt(func, ramCoef, align=32, padding=45, scale=1, oShape=None, **_):
  opt = Option()
  opt.modelCached = func
  opt.ramCoef = ramCoef
  opt.align = align
  opt.padding = padding
  opt.scale = scale
  opt.squeeze = identity
  opt.unsqueeze = identity
  opt.oShape = oShape
  return opt

def getOptS(modelPath, modules, ramCoef):
  opt = Option(modelPath)
  weights = getStateDict(modelPath)
  opt.modules = modules
  opt.ramOffset = config.getRunType() * len(modules)
  for i, key in enumerate(modules):
    m = modules[key]
    wKey = m['weight']
    constructor = m.get('f', 0)
    rc = m['ramCoef'][config.getRunType()] if 'ramCoef' in m else ramCoef[opt.ramOffset + i]
    o = dict((k, m[k]) for k in ('align', 'padding', 'scale') if k in m)
    model = initModel(opt, weights[wKey], key, constructor, args=m.get('args', []))
    if 'outShape' in m:
      opt.__dict__[key] = newOpt(model, rc / len(modules), **o)
    else:
      model.ramCoef = rc
      opt.__dict__[key] = model
  return opt

def setOutShape(opt, height, width):
  load = width * height
  od = opt.__dict__
  freeMem = config.calcFreeMem(1 / len(opt.modules))
  for key, o in opt.modules.items():
    batchSize = opt.bf(load, od[key].ramCoef, freeMem)
    if 'outShape' in o:
      q = o['outShape']
      od[key].outShape = [batchSize, *q[1:-2], int(height * q[-2]), int(width * q[-1])]
      if 'staticDims' in o:
        for i in o['staticDims']:
          od[key].outShape[i] = q[i]
    if 'streams' in o and (not 0 in o.get('staticDims', {})):
      for name in o['streams']:
        od[name].send((None, batchSize))
  return opt

def getOptP(opt, bf=getBatchSize):
  opt.startPadding = 0
  opt.i = 0
  opt.currentSize = 0
  opt.outStart = 0
  opt.outEnd = 0
  opt.bf = bf
  return opt

extendRes = lambda res, item: res.extend(item) if type(item) == list else (None if item is None else res.append(item))
def makeStreamFunc(func, node, opt, nodes, name, padStates, initFunc, putFunc):
  for n in nodes:
    node.append(n)
  def f(x):
    node.reset()
    node.trace(0, p='{} start'.format(name))

    if not opt.currentSize and not x is None:
      opt.currentSize = initFunc(opt, x)
    if opt.i % 31 == 0:
      setOutShape(opt, *opt.currentSize)

    if opt.end:
      for s in padStates:
        s.setPadding(opt.end)
      opt.end = 0
    if opt.start:
      opt.startPadding = opt.start
      for s in padStates:
        s.setPadding(opt.start)
      opt.start = 0
    out = []
    last = True if x is None else None
    if not last:
      putFunc(opt.pad(x.unsqueeze(0)))
      opt.i += 1
      extend(out, opt.out.send(None))
    while last:
      try:
        extend(out, opt.out.send(last))
      except StopIteration: break
    if x is None and opt.outEnd:
      out = out[:opt.outEnd]
      opt.outEnd = 0
    l = len(out)
    out = out[opt.outStart:]
    opt.outStart = max(0, opt.outStart - l)
    node.trace(len(out))
    res = []
    for item in out:
      extendRes(res, func(opt.unpad(item)))
    return res
  return f