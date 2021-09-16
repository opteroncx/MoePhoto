import torch
t = torch.arange(10)

config = type('', (), {'device': lambda *_: torch.device('cpu')})()

def pp(name, *ts):
  ss = [x.shape for x in ts]
  print(name, *ss)
  return ts[0]

identity = lambda x, *_: x

from typing import Callable, List, Union

def DefaultStreamSource(last=None):
  while not last:
    t = yield
    last = t[0] if type(t) == tuple else t

class StreamState():
  def __init__(self, window=None, device=config.device(), offload=False, store=True, tensor=True, name=None, **_):
    self.source = DefaultStreamSource()
    next(self.source)
    self.wm1 = window - 1 if window else 0
    self.device = device
    self.offload = tensor and offload
    self.store = store
    self.batchFunc = torch.stack if tensor else identity
    self.name = name
    self.start = 0
    self.end = 0
    self.state = []

  def clear(self):
    self.state.clear()

  def getSize(self, size=None):
    if self.start:
      self.start -= self.pad(self.start)
    ls = len(self.state)
    if ls < self.wm1 + (size or 1):
      return 0
    lb = ls - self.wm1
    return min(size, lb) if size else lb

  def popBatch(self, size=1, last=None, *_, **__):
    if last and self.end:
      self.end -= self.pad(self.end)
    r = self.getSize(size)
    if not r:
      return
    batch = [self.batchFunc(self.state[i:i + self.wm1 + 1]) for i in range(r)] if self.wm1 else self.state[:r]
    self.state = self.state[r:]
    batch = self.batchFunc(batch)
    return batch.to(self.device) if self.offload else batch

  def setPadding(self, padding):
    if padding > 0: self.start = padding
    elif padding < 0: self.end = padding

  def pad(self, padding: int, *_, **__):
    if padding == 0:
      return 0
    absPad = abs(padding)
    size = 1 + absPad * 2
    if len(self.state) < size:
      return 0
    offset = padding - 2 if padding < 0 else 0
    ids = (torch.arange(absPad, 0, -1) + padding + offset).tolist()
    batch = [self.state[i] for i in ids]
    self.state = (self.state + batch) if padding < 0 else (batch + self.state)
    return padding

  def push(self, batch: Union[torch.Tensor, List[torch.Tensor]], *_, **__):
    if batch is None:
      return
    if self.offload:
      batch = [t.cpu() for t in batch if t != None] if type(batch) == list else batch.cpu()
    self.store and self.state.extend(t for t in batch)
    return batch

  def bind(self, stateIter):
    self.source = stateIter
    return self.source

  def __len__(self):
    return self.getSize()

  def __str__(self):
    return 'StreamState {}'.format(self.name) if self.name else 'anonymous StreamState'

  def pull(self, last=None, size=None):
    t = (last, size) if size else last
    try:
      self.source.send(t)
      return 1
    except StopIteration: return

  @classmethod
  def run(_, f: Callable, states, size: int, args=[], last=None, pipe=False):
    t = yield
    while True:
      last, size = t if type(t) == tuple else (t, size)
      r = min(s.getSize() for s in states)
      if not r or (r < size and not last):
        if not pipe:
          break
        if not any(s.pull(last) for s in states): # every source is drained
          pipe = False # try to check sources' size for last time
        if not last:
          t = yield
        continue
      nargs = list(args) + [s.popBatch(min(r, size), last) for s in states]
      out = f(*nargs)
      t = yield out

  @classmethod
  def pipe(cls, f: Callable, states, targets, size=1, args=[]):
    it = cls.run(f, states, size, args, pipe=True)
    next(it)
    itPipe = pipeFunc(it, targets, size)
    next(itPipe)
    for t in targets:
      t.bind(itPipe)
    return itPipe

def pipeFunc(it, targets, size, last=False):
  t = yield
  while True:
    flag, size = t if type(t) == tuple else (t, size)
    last |= bool(flag)
    try:
      out = it.send((last, size))
      for t in targets:
        t.push(out)
      t = yield out
    except StopIteration: break

class KeyFrameState():
  def __init__(self, window):
    self.window = window
    self.count = 0
    self.last = None

  def getSize(self, size=1 << 30):
    return size

  def pull(self, last=None):
    return not last

  def popBatch(self, size=1, last=None, *_, **__):
    res = torch.zeros((size,), dtype=torch.bool)
    for i in range(-self.count % self.window, size, self.window):
      res[i] = True
    res[-1] = bool(last)
    self.count += size
    return res

def calcKeyframeFeature(window):
  print('calcKeyframeFeature', len(window), window[0])
  return window
def getKeyframeFeature(keyframe, isKeyFrame):
  print('getKeyframeFeature', len(keyframe))
  return [(calcKeyframeFeature(w) if b else None) for w, b in zip(keyframe, isKeyFrame)]
def calcBackward(inp, flowInp, keyframeFeature):
  n = inp.shape[0]
  print('calcBackward', n, last)
  # flowInp = self.get_backward_flow(flowInp[:-1] if last else flowInp) [[x_i, x_i+1], ...]
  feat_prop = inp.new_zeros(1, 1) # batch, channel, height, width
  out = [feat_prop]
  if last: # require at least 2 backward reference frames
    out = out * 3 # pad 2 empties for the last window
  for i in range(n - 1, -1, -1):
    feat_prop = out[0]
    if i < n - 1 and not last:
      # feat_prop = self.flow_warp(feat_prop, flowInp[i])
      pass
    if keyframeFeature[i] != None:
      # feat_prop = torch.cat([feat_prop, keyframeFeature[i]], dim=1)
      # feat_prop = self.backward_fusion(feat_prop)
      pass
    # feat_prop = torch.cat([inp[i], feat_prop], dim=1)
    # feat_prop = self.backward_trunk(feat_prop)
    feat_prop = inp[i] + feat_prop
    out.insert(0, feat_prop)
  return [out[i:i+3] for i in range(len(out) - 3)] # only window[0] is used
def calcFlowForward(flowInp):
  out = []
  if flowForward.first:
    out.append(None)
    flowInp = flowInp[1:]
    flowForward.first = 0
  # flowInp = self.get_forward_flow(flowInp) [[x_i, x_i+1], ...]
  out.extend(list(flowInp))
  return out
def calcForward(inp, flowInp, keyframeFeature, backward):
  n = inp.shape[0]
  print('calcBackward', n)
  feat_prop = inp.new_zeros(1, 1) # batch, channel, height, width
  out = []
  for i in range(n):
    if flowInp[i] != None:
      # feat_prop = self.flow_warp(feat_prop, flowInp[i])
      pass
    if keyframeFeature[i] != None:
      # feat_prop = torch.cat([feat_prop, keyframeFeature[i]], dim=1)
      # feat_prop = self.forward_fusion(feat_prop)
      pass
    feat_prop = inp[i] + feat_prop
    out.append(feat_prop)
    # feat_prop = torch.cat([inp[i], backward[i][0], feat_prop], dim=1)
    # feat_prop = self.forward_trunk(feat_prop)
  return out

inp = StreamState()
inp1 = StreamState()
inp2 = StreamState()
backwardInp = StreamState()
flowInp = StreamState(2)
flowForwardInp = StreamState()
flowForwardInp.setPadding(1)
flowBackwardInp = StreamState()
flowBackwardInp.setPadding(-1)
isKeyFrame = KeyFrameState(7)
keyframeFeatureInp = StreamState(7, tensor=False)
keyframeFeatureInp.setPadding(2)
keyframeFeatureInp.setPadding(-3)
isKeyFrame1 = StreamState(tensor=False)
isKeyFrame2 = StreamState(tensor=False)
isKeyFrame3 = StreamState(tensor=False)
StreamState.pipe(pp, [inp], [inp1, inp2, keyframeFeatureInp, flowInp, backwardInp], args=['input fan out'])
StreamState.pipe(pp, [flowInp], [flowForwardInp, flowBackwardInp], args=['flow fan out'])
StreamState.pipe(pp, [isKeyFrame], [isKeyFrame1, isKeyFrame2, isKeyFrame3], args=['keyframe fan out'])
keyframeFeature = StreamState(tensor=False)
StreamState.pipe(getKeyframeFeature, [keyframeFeatureInp, isKeyFrame1], [keyframeFeature], size=3)
keyframeFeature1 = StreamState(tensor=False)
keyframeFeature2 = StreamState(tensor=False)
StreamState.pipe(pp, [keyframeFeature], [keyframeFeature1, keyframeFeature2], args=['keyframe feature fan out'])
backward = StreamState(3)
StreamState.pipe(calcBackward, [backwardInp, flowBackwardInp, keyframeFeature1], [backward], size=20)
flowForward = StreamState(tensor=False)
flowForward.first = 1
StreamState.pipe(calcFlowForward, [flowForwardInp], [flowForward], size=4)
forward = StreamState()
StreamState.pipe(calcForward, [inp1, flowForward, keyframeFeature2, backward], [forward])
upsample = StreamState()
out = StreamState.pipe(pp, [inp2, forward], [upsample], args=['upsample'])
last = None
inp.push(t)