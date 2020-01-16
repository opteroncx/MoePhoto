import time
import json
from os.path import exists
from functools import partial
from gevent import spawn

ops = {}
loadedOps = {}
needSave = False
noNotify = { 'toFloat', 'toOutput', 'Channel', 'toBuffer', 'toTorch' }

def recurse(f):
  def r(node):
    f(node)
    for n in node.nodes:
      r(n)
  return r
getNodeETA = lambda node: ops[node.op].weight * node.load * max(0, node.total - node.gone)
sumETT = lambda node: sum(map(lambda n: n.ett, node.nodes)) if len(node.nodes) else 1
getETT = lambda node: ops[node.op].weight * node.load * max(0, node.total) * sumETT(node)
def updateNode(node):
  s = ops[node.op].weight * node.load * sumETT(node)
  if node.total >= 0:
    node.ett = node.total * s
    node.eta = (node.total - node.gone) * s
  else:
    node.ett = node.eta = -1
slideAverage = lambda coef: lambda op, sample: coef * op.weight + (1 - coef) * sample
setNodeCallback = lambda node, callback, any, bench: node.setCallback(callback, bench) if any or hasattr(node, 'name') else None
setCallback = lambda node, callback, all=False, bench=False: recurse(lambda node: setNodeCallback(node, callback, all, bench))(node)
getOpKey = lambda op: hash(frozenset(op.items()))
NullFunc = lambda *args: None
loadOps = lambda path: spawn(loadInternal, path).start()
serializeOp = lambda op: dict(op=op.op, weight=op.weight, samples=op.samples)
serializeOps = lambda: [serializeOp(ops[key]) for key in ops]

def saveInternal(path):
  with open(path, 'w') as fp:
    json.dump(serializeOps(), fp, ensure_ascii=False, indent=2)

def saveOps(path=None, force=False):
  global needSave
  if path and (needSave or force):
    spawn(saveInternal, path).start()
    needSave = False
  return serializeOps()

def loadInternal(path):
  if not exists(path):
    return
  with open(path, 'r') as fp:
    res = json.load(fp)
  for op in res:
    loadedOps[getOpKey(op['op'])] = (op['weight'], op['samples'])

def initOp(op, learn=True):
  op.weight = 1e-6 if learn else 1
  op.samples = 0

def clearOps(node, flag=True):
  if flag:
    loadedOps.clear()
    recurse(lambda n: initOp(ops[n.op], n.learn))(node)

def newOp(learn, define={}, updater=slideAverage(.9)):
  def op():pass
  key = getOpKey(define)
  op.op = define
  if key in loadedOps:
    op.weight = loadedOps[key][0]
    op.samples = loadedOps[key][1]
  else:
    initOp(op, learn)
  def f(sample):
    global needSave
    if not op.samples:
      needSave = True
    op.samples += 1
    op.weight = updater(op, sample) if op.samples > 1 else sample
  op.update = f
  return op

def updateAncestor(node, eta=False):
  p = node.parent
  while p:
    i = p.nodes.index(node)
    updateNode(p)
    if eta and p.total >= 0:
      p.eta += node.eta - sum(map(lambda n: n.ett, p.nodes[:i + 1]))
      if p.eta < 0:
        p.eta = p.ett * (p.total - p.gone) / p.total
    node = p
    p = p.parent

def initialETA(node):
  node.gone = 0
  s = sum(map(initialETA, node.nodes)) if len(node.nodes) else 1
  c = getNodeETA(node)
  node.eta = c * s if node.total >= 0 else -1
  node.ett = node.eta
  return node.ett

class Node():
  def __init__(self, op, load=1, total=1, learn=30, callback=NullFunc, name=None):
    self.load = load
    self.total = total
    self.gone = 0
    self.ett = 0
    self.eta = 0
    self.parent = None
    self.bench = False
    self.learn = learn or 0
    self.callback = callback
    self.nodes = []
    key = getOpKey(op)
    self.op = key
    if name:
      self.name = name
    if not key in ops:
      ops[key] = newOp(learn, op)

  def append(self, child):
    self.nodes.append(child)
    child.parent = self
    return self

  def setCallback(self, callback=NullFunc, bench=False):
    self.callback = NullFunc if ops[self.op].op.get('op', '') in noNotify else callback
    self.bench = bench and self.learn
    if self.bench:
      self.learn = float('inf')

  def multipleLoad(self, scale=1):
    if len(self.nodes):
      for node in self.nodes:
        node.multipleLoad(scale)
    else:
      self.load *= scale

  def reset(self):
    self.gone = 0
    self.ett = getETT(self)
    self.eta = self.ett
    if self.learn:
      self.mark = time.perf_counter()
    return self

  def trace(self, progress=1, **kwargs):
    global needSave
    self.gone += progress
    op = ops[self.op]
    if self.learn > op.samples and progress > 0:
      mark = time.perf_counter()
      delta = mark - self.mark
      op.update(delta / self.load / progress)
      if op.samples >= self.learn:
        self.learn = False
        needSave = True
      self.mark = mark
      if self.bench:
        kwargs.update(serializeOp(op))
    if progress > 0:
      updateNode(self)
      updateAncestor(self, True)
    return self.callback(self, kwargs)

  def bindFunc(self, f):
    def g(*args, **kwargs):
      self.reset()
      self.trace(0)
      res = f(*args, **kwargs)
      self.trace()
      return res
    return g

  def update(self, content):
    if 'op' in content:
      content['op'] = getOpKey(content['op'])
    self.__dict__.update(content)
    updateNode(self)
    updateAncestor(self)

  def remove(self, update=False):
    self.parent.nodes.remove(self)
    p = self.parent
    self.parent = None
    if update:
      updateNode(p)
      updateAncestor(p)

  def moveTo(self, target, pos=-1):
    flag = self.parent != target
    if self.parent:
      self.remove(flag)
    if pos < 0:
      target.append(self)
    else:
      target.nodes.insert(pos, self)
      self.parent = target
    if flag:
      updateAncestor(self)

  def toStop(self):
    self.total = self.gone + 1
    return self.trace(0)