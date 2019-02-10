import time
from functools import partial

ops = {}

def recurse(f):
  def r(node):
    f(node)
    for n in node.nodes:
      r(n)
  return r
getNodeETA = lambda node: ops[node.op].weight * node.load * (node.total - node.gone)
sumETT = lambda node: sum(map(lambda n: n.ett, node.nodes)) if len(node.nodes) else 1
getETT = lambda node: ops[node.op].weight * node.load * node.total * sumETT(node)
def updateNode(node):
  s = ops[node.op].weight * node.load * sumETT(node)
  node.ett = node.total * s
  node.eta = (node.total - node.gone) * s
slideAverage = lambda coef: lambda op, sample: coef * op.weight + (1 - coef) * sample
setNodeCallback = lambda node, callback, any: node.setCallback(callback) if any or hasattr(node, 'name') else None
setCallback = lambda node, callback, all=False: recurse(lambda node: setNodeCallback(node, callback, all))(node)
getOpKey = lambda op: hash(frozenset(op.items()))
NullFunc = lambda *args: None

def newOp(updater=slideAverage(.5)):
  def op():pass
  op.weight = 1
  op.samples = 0
  def f(sample):
    op.samples += 1
    op.weight = updater(op, sample) if op.samples > 1 else sample
  op.update = f
  return op

def updateAncestor(node, eta=False):
  p = node.parent
  while p:
    i = 0
    while not (p.nodes[i] is node):
      i += 1
    updateNode(p)
    if eta:
      p.eta += node.eta - sum(map(lambda n: n.ett, p.nodes[:i + 1]))
      if p.eta < 0:
        p.eta = 0
    node = p
    p = p.parent

def initialETA(node):
  node.gone = 0
  c = getNodeETA(node)
  if len(node.nodes):
    node.eta = c * sum(map(initialETA, node.nodes))
  else:
    node.eta = c
  node.ett = node.eta
  return node.ett

class Node():
  def __init__(self, op, load=1, total=1, learn=3, callback=NullFunc, name=None):
    self.load = load
    self.total = total
    self.gone = 0
    self.ett = 0
    self.eta = 0
    self.parent = None
    self.learn = learn
    self.callback = callback
    self.nodes = []
    key = getOpKey(op)
    self.op = key
    if name:
      self.name = name
    if not key in ops:
      ops[key] = newOp()
    else:
      if not learn or ops[key].samples >= learn:
        self.learn = False

  def append(self, child):
    self.nodes.append(child)
    child.parent = self
    return self

  def setCallback(self, callback=NullFunc):
    self.callback = callback

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
    self.gone += progress
    if self.learn and progress > 0:
      mark = time.perf_counter()
      delta = mark - self.mark
      ops[self.op].update(delta / self.load / progress)
      self.mark = mark
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