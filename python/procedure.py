# pylint: disable=E1101
from functools import reduce
import numpy as np
from config import config
from progress import Node
from imageProcess import (
  toFloat, toOutput, toOutput8, toTorch, toNumPy, toBuffer,
  readFile, writeFile,
  BGR2RGB, BGR2RGBTorch, RGBFilter,
  resize, restrictSize, windowWrap,
  apply, identity, previewFormat, previewPath
)
import runSR
import runDN
import dehaze
import runSlomo
import videoSR
import ESTRNN
from worker import context

videoOps = dict(slomo=runSlomo.WindowSize, VSR=videoSR.WindowSize, demob=ESTRNN.WindowSize)
applyNonNull = lambda v, f: NonNullWrap(f)(v)
NonNullWrap = lambda f: lambda x: f(x) if not x is None else None
newNode = lambda opt, op, load=1, total=1: Node(op, load, total, name=opt.get('name', None))

def convertValues(T, o, keys):
  for key in keys:
    if key in o:
      o[key] = T(o[key])

def appendFuncs(f, node, funcs, wrap=True):
  g = node.bindFunc(f)
  funcs.append(NonNullWrap(g) if wrap else g)
  return node

fPreview = [
  0,
  toFloat,
  toOutput8,
  (lambda im: im.astype(np.uint8)),
  0,
  lambda im: writeFile(im, context.shared, context, previewFormat),
  lambda *_: context.root.trace(0, preview=previewPath, fileSize=context.shared.tell())]
funcPreview = lambda im: reduce(applyNonNull, fPreview, im)

def procInput(source, bitDepth, fs, out):
  out['load'], out['sf']  = 1, 1
  node = Node({'op': 'toTorch', 'bits': bitDepth})
  fs.append(NonNullWrap(node.bindFunc(toTorch(bitDepth, config.dtype(), config.device()))))
  return fs, [node], out

def procDN(opt, out, *_):
  DNopt = opt['opt']
  node = newNode(opt, dict(op='DN', model=opt['model']), out['load'])
  return [NonNullWrap(node.bindFunc(RGBFilter(DNopt)))], [node], out

def convertChannel(out):
  out['channel'] = 0
  fs=[]
  return fs, [appendFuncs(BGR2RGBTorch, Node(dict(op='Channel')), fs)]

def procSR(opt, out, *_):
  load = out['load']
  scale = opt['scale']
  mode = opt['model']
  SRopt = opt['opt']
  es = SRopt.ensemble + 1
  if not scale > 1:
    raise TypeError('Invalid scale setting for SR.')
  out['load'] = load * scale * scale
  fs, ns = convertChannel(out) if out['channel'] and mode == 'gan' else ([], [])
  ns.append(appendFuncs(runSR.sr(SRopt), newNode(opt, dict(op='SR', model=mode, scale=scale), load * es), fs))
  return fs, ns, out

def procVSR(opt, out, *_):
  load = out['load']
  scale = 4
  out['load'] = load * scale * scale
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  ns.append(newNode(opt, dict(op='VSR', learn=0), load))
  return fs + [videoSR.doVSR], ns, out

def procSlomo(opt, out, *_):
  load = out['load']
  out['sf'] *= opt['sf']
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  node = newNode(opt, dict(op='slomo'), load, opt['sf'])
  return fs + [runSlomo.doSlomo], ns + [node], out

def procDemob(opt, out, *_):
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  ns.append(newNode(opt, dict(op='ESTRNN', learn=0), out['load']))
  return fs + [ESTRNN.doESTRNN], ns, out

def procDehaze(opt, out, *_):
  load = out['load']
  dehazeOpt = opt['opt']
  model = opt.get('model', 'dehaze')
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  node = newNode(opt, dict(op=model), load)
  ns.append(appendFuncs(RGBFilter(dehazeOpt), node, fs))
  return fs, ns, out

def procResize(opt, out, nodes):
  load = out['load']
  node = newNode(opt, dict(op='resize', mode=opt['method']), load)
  return [node.bindFunc(NonNullWrap(resize(opt, out, len(nodes), nodes)))], [node], out

def procOutput(opt, out, *_):
  load = out['load']
  node0 = Node(dict(op='toFloat'), load)
  bitDepthOut = out['bitDepth']
  node1 = newNode(opt, dict(op='toOutput', bits=bitDepthOut), load)
  fOutput = node1.bindFunc(toOutput(bitDepthOut))
  fTrace = lambda x: context.root.trace(1 / out['sf']) or x
  fs = [NonNullWrap(node0.bindFunc(toFloat)), NonNullWrap(fOutput)]
  ns = [node0, node1]
  if out['source']:
    fPreview[0] = restrictSize(2048)
    fs1 = [node0.bindFunc(toFloat), fOutput]
    if previewFormat:
      def o(im):
        res = reduce(applyNonNull, fs1, im)
        funcPreview(im)
        return [res]
    else:
      o = lambda im: [reduce(applyNonNull, fs1, im)]
    fs = [o, fTrace]
    if out['channel']:
      fPreview[4] = BGR2RGB
    else:
      fPreview[4] = identity
      ns.append(appendFuncs(BGR2RGB, Node(dict(op='Channel')), fs1, False))
      out['channel'] = 1
    ns.append(appendFuncs(toBuffer(bitDepthOut), Node(dict(op='toBuffer', bits=bitDepthOut), load), fs1, False))
  return fs, ns, out

procs = dict(
  file=(lambda _, _0, nodes:
    procInput('file', 8, [context.getFile, readFile(nodes, context)], dict(bitDepth=8, channel=0, source=0))),
  buffer=(lambda opt, *_:
    procInput('buffer', opt['bitDepth'], [toNumPy(opt['bitDepth'])], dict(bitDepth=opt['bitDepth'], channel=1, source=1))),
  DN=procDN, SR=procSR, output=procOutput, slomo=procSlomo,
  dehaze=procDehaze, resize=procResize, VSR=procVSR, demob=procDemob
  )

stepOpts = dict(
  SR={'toInt': ['scale', 'ensemble'], 'getOpt': runSR},
  resize={'toInt': ['width', 'height'], 'toFloat': ['scaleW', 'scaleH']},
  DN={'toFloat': ['strength'], 'getOpt': runDN},
  dehaze={'toFloat': ['strength'], 'getOpt': dehaze},
  slomo={'toInt': ['sf'], 'getOpt': runSlomo},
  VSR={'getOpt': videoSR},
  demob={'getOpt': ESTRNN}
)
def genProcess(steps, root=True, outType=None):
  funcs=[]
  nodes=[]
  last = identity
  rf = lambda im: reduce(apply, funcs, im)
  if root:
    stepOffset = 0 if steps[0]['op'] == 'file' else 2
    for i, opt in enumerate(steps):
      opt['name'] = i + stepOffset
      if opt['op'] in stepOpts:
        stepOpt = stepOpts[opt['op']]
        convertValues(int, opt, stepOpt.get('toInt', []))
        convertValues(float, opt, stepOpt.get('toFloat', []))
        if 'getOpt' in stepOpt:
          opt['opt'] = stepOpt['getOpt'].getOpt(opt)
    if steps[-1]['op'] != 'output':
      steps.append(dict(op='output'))
    config.getFreeMem(True)
    process = lambda im, name=None: last(rf(im), name, context)
  else:
    process = rf
  for i, opt in enumerate(steps):
    op = opt['op']
    fs, ns, outType = procs[op](opt, outType, nodes)
    funcs.extend(fs)
    nodes.extend(ns)
    if op in videoOps:
      if i + 1 < len(steps):
        f, nodesAfter = genProcess(steps[i + 1:], False, outType)
      else:
        f = identity
        nodesAfter = []
      videoOpt = opt['opt']
      func = funcs[-1](f, nodes[-1], videoOpt)
      funcs[-1] = windowWrap(func, videoOpt, videoOps[op]) if videoOps[op] > 1 else func
      nodeAfter = Node({}, total=opt.get('sf', 1), learn=0)
      for node in nodesAfter:
        nodeAfter.append(node)
      nodes.append(nodeAfter)
      break
  if root and steps[0]['op'] == 'file':
    n = Node({'op': 'write'}, outType['load'])
    nodes.append(n)
    last = n.bindFunc(writeFile)
  else:
    context.imageMode = 'RGB'
  return process, nodes