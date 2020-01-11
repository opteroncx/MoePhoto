# pylint: disable=E1101
from functools import reduce
import numpy as np
from config import config
from progress import Node
from imageProcess import toFloat, toOutput, toOutput8, toTorch, toNumPy, toBuffer, toInt, readFile, writeFile, BGR2RGB, BGR2RGBTorch, resize, restrictSize, windowWrap, apply, identity, previewFormat, previewPath, ensemble
import runDN
import runSR
import runSlomo
import dehaze
from worker import context

applyNonNull = lambda v, f: NonNullWrap(f)(v)
NonNullWrap = lambda f: lambda x: f(x) if not x is None else None

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
  lambda im: writeFile(im, context.shared, previewFormat),
  lambda *_: context.root.trace(0, preview=previewPath, fileSize=context.shared.tell())]
funcPreview = lambda im: reduce(applyNonNull, fPreview, im)

def procInput(source, bitDepth, fs, out):
  out['load'] = 1
  node = Node({'op': 'toTorch', 'bits': bitDepth})
  fs.append(NonNullWrap(node.bindFunc(toTorch(bitDepth, config.dtype(), config.device()))))
  return fs, [node], out

def procDN(opt, out, *_):
  DNopt = opt['opt']
  node = Node(dict(op='DN', model=opt['model']), out['load'])
  if 'name' in opt:
    node.name = opt['name']
  return [NonNullWrap(node.bindFunc(ensemble(DNopt)))], [node], out

def convertChannel(out):
  out['channel'] = 0
  fs=[]
  return fs, [appendFuncs(BGR2RGBTorch, Node(dict(op='Channel')), fs)]

def procSR(opt, out, *_):
  load = out['load']
  scale = opt['scale']
  mode = opt['model']
  SRopt = opt['opt']
  if not scale > 1:
    raise TypeError('Invalid scale setting for SR.')
  out['load'] = load * scale * scale
  fs, ns = convertChannel(out) if out['channel'] and mode == 'gan' else ([], [])
  ns.append(appendFuncs(runSR.sr(SRopt), Node(dict(op='SR', model=mode), load), fs))
  if 'name' in opt:
    ns[-1].name = opt['name']
  return fs, ns, out

def procSlomo(opt, out, *_):
  load = out['load']
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  node = Node(dict(op='slomo'), load, opt['sf'], name=opt['name'] if 'name' in opt else None)
  return fs + [runSlomo.doSlomo], ns + [node], out

def procDehaze(opt, out, *_):
  load = out['load']
  dehazeOpt = opt['opt']
  model = opt.get('model', 'dehaze')
  fs, ns = convertChannel(out) if out['channel'] else ([], [])
  node = Node(dict(op=model), load, name=opt['name'] if 'name' in opt else None)
  ns.append(appendFuncs(lambda im: dehaze.Dehaze(dehazeOpt, im), node, fs))
  return fs, ns, out

def procResize(opt, out, nodes):
  node = Node(dict(op='resize', mode=opt['method']), 1, name=opt['name'] if 'name' in opt else None)
  return [node.bindFunc(NonNullWrap(resize(opt, out, len(nodes), nodes)))], [node], out

def procOutput(opt, out, *_):
  load = out['load']
  node0 = Node(dict(op='toFloat'), load)
  bitDepthOut = out['bitDepth']
  node1 = Node(dict(op='toOutput', bits=bitDepthOut), load, name=opt['name'] if 'name' in opt else None)
  fOutput = node1.bindFunc(toOutput(bitDepthOut))
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
    fs = [o]
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
    procInput('file', 8, [context.getFile, readFile(nodes)], dict(bitDepth=8, channel=0, source=0))),
  buffer=(lambda opt, *_:
    procInput('buffer', opt['bitDepth'], [toNumPy(opt['bitDepth'])], dict(bitDepth=opt['bitDepth'], channel=1, source=1))),
  DN=procDN, SR=procSR, output=procOutput, slomo=procSlomo, dehaze=procDehaze, resize=procResize
  )

def genProcess(steps, root=True, outType=None):
  funcs=[]
  nodes=[]
  last = identity
  rf = lambda im: reduce(apply, funcs, im)
  if root:
    for i, opt in enumerate(steps):
      opt['name'] = i
    for opt in filter((lambda opt: opt['op'] == 'SR'), steps):
      toInt(opt, ['scale', 'ensemble'])
      opt['opt'] = runSR.getOpt(opt)
    for opt in filter((lambda opt: opt['op'] == 'resize'), steps):
      toInt(opt, ['width', 'height'])
      if 'scaleW' in opt:
        opt['scaleW'] = float(opt['scaleW'])
      if 'scaleH' in opt:
        opt['scaleH'] = float(opt['scaleH'])
    for opt in filter((lambda opt: opt['op'] == 'DN'), steps):
      opt['opt'] = runDN.getOpt(opt)
    for opt in filter((lambda opt: opt['op'] == 'dehaze'), steps):
      opt['opt'] = dehaze.getOpt(opt)
    slomos = [*filter((lambda opt: opt['op'] == 'slomo'), steps)]
    for opt in slomos:
      toInt(opt, ['sf'])
      opt['opt'] = runSlomo.getOpt(opt)
    if len(slomos):
      slomos[-1]['opt'].notLast = 0
    if steps[-1]['op'] != 'output':
      steps.append(dict(op='output'))
    config.getFreeMem(True)
    process = lambda im, name=None: last(rf(im), name)
  else:
    process = rf
  for i, opt in enumerate(steps):
    op = opt['op']
    fs, ns, outType = procs[op](opt, outType, nodes)
    funcs.extend(fs)
    nodes.extend(ns)
    if op == 'slomo':
      if i + 1 < len(steps):
        f, nodesAfter = genProcess(steps[i + 1:], False, outType)
      else:
        f = identity
        nodesAfter = []
      slomoOpt = opt['opt']
      slomo = funcs[-1](f, nodes[-1], slomoOpt)
      funcs[-1] = windowWrap(slomo, slomoOpt, 2)
      nodeAfter = Node({}, total=opt['sf'], learn=0)
      for node in nodesAfter:
        nodeAfter.append(node)
      nodes.append(nodeAfter)
      break
  if root and steps[0]['op'] == 'file':
    n = Node({'op': 'write'}, outType['load'])
    nodes.append(n)
    last = n.bindFunc(writeFile)
  return process, nodes