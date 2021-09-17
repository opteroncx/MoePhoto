import torch

config = type('', (), {'device': lambda *_: torch.device('cpu')})()

def pp(name, *ts, **_):
  print(name, *ts)
  return ts[0]

identity = lambda x, *_: x

from typing import Callable, List, Union

def DefaultStreamSource(last=None):
  flag = True
  while flag:
    t = yield
    last = t[0] if type(t) == tuple else t
    flag &= not last

class StreamState():
  def __init__(self, window=None, device=config.device(), offload=False, store=True, tensor=True, name=None, reserve=0, **_):
    assert True if window is None else window > 0
    assert tensor or offload == False
    assert reserve >= 0
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
    self.reserve = [0 for _ in range(reserve)]
    self.state = []

  def clear(self):
    self.state.clear()

  def getSize(self, size=None):
    if self.start:
      self.start -= self.pad(self.start)
      if self.start:
        return 0
    ls = len(self.state)
    if ls < self.wm1 + (size or 1):
      return 0
    lb = ls - self.wm1
    return min(size, lb) if size else lb

  def popBatch(self, size=1, **_):
    r = self.getSize(size)
    if not r:
      return
    batch = [self.batchFunc(self.state[i:i + self.wm1 + 1]) for i in range(r)] if self.wm1 else self.state[:r]
    self.reserve = self.state[r - len(self.reserve):r]
    self.state = self.state[r:]
    batch = self.batchFunc(batch)
    return batch.to(self.device) if self.offload else batch

  def setPadding(self, padding):
    if padding > 0: self.start = padding
    elif padding < 0: self.end = padding
    return self

  def pad(self, padding: int, *_, **__):
    if padding == 0:
      return 0
    absPad = abs(padding)
    size = 1 + absPad * 2
    if len(self.reserve) + len(self.state) < size:
      return 0
    s = self.reserve + self.state
    offset = padding - 2 if padding < 0 else 0
    ids = (torch.arange(absPad, 0, -1) + padding + offset).tolist()
    batch = [s[i] for i in ids]
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
    flag = None
    t = (last, size) if size else last
    try:
      self.source.send(t)
      flag = 1
    except StopIteration: pass
    if last:
      tuple(self.source) # drain source
      flag = None
    if last and self.end:
      self.end -= self.pad(self.end)
      assert not self.end
    return flag

  @classmethod
  def run(_, f: Callable, states, size: int, args=[], last=None, pipe=False):
    t = yield
    while True:
      last, size = t if type(t) == tuple else (t, size)
      r = min(s.getSize() for s in states)
      if not r or (r < size and not last):
        if not pipe:
          break
        pr = [s.pull(last) for s in states]
        pipe &= any(pr) # can end if every source is drained
        t = yield
        continue
      upLast = last and r <= size
      nargs = list(args) + [s.popBatch(min(r, size), last=upLast) for s in states]
      out = f(*nargs, last=upLast)
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
  def __init__(self, window, count=0):
    self.window = window
    self.count = count

  def getSize(self, size=1 << 30):
    return size

  def pull(self, last=None, *_, **__):
    return not last

  def popBatch(self, size=1, last=None):
    res = torch.zeros((size,), dtype=torch.bool)
    for i in range(-self.count % self.window, size, self.window):
      res[i] = True
    if last:
      print('KeyFrameState last pop', self.count, size)
      res[-1] = True
    self.count += size
    return res

def calcKeyframeFeature(window):
  print('calcKeyframeFeature', len(window), window[0])
  return window
def getKeyframeFeature(keyframe, isKeyFrame, **_):
  print('getKeyframeFeature', isKeyFrame)
  return [(calcKeyframeFeature(w) if b else None) for w, b in zip(keyframe, isKeyFrame)]
def calcFlowBackward(flowInp, last):
  out = []
  # flowInp = self.get_backward_flow(flowInp) [[x_i, x_i+1], ...]
  out = list(flowInp)
  if last:
    out.append(None)
    print('calcFlowBackward last', out)
  return out
def calcBackward(inp, flowInp, keyframeFeature, last):
  n = inp.shape[0]
  print('calcBackward', last, flowInp, keyframeFeature)
  feat_prop = inp.new_zeros(1, 1) # batch, channel, height, width
  out = [feat_prop]
  if last: # require at least 2 backward reference frames
    out = out * 3 # pad 2 empties for the last window
  for i in range(n - 1, -1, -1):
    feat_prop = out[0]
    if i < n - 1 or not last:
      # feat_prop = self.flow_warp(feat_prop, flowInp[i])
      pass
    else:
      assert flowInp[i] == None
    if keyframeFeature[i] != None:
      # feat_prop = torch.cat([feat_prop, keyframeFeature[i]], dim=1)
      # feat_prop = self.backward_fusion(feat_prop)
      assert keyframeFeature[i][3] == inp[i]
    # feat_prop = torch.cat([inp[i], feat_prop], dim=1)
    # feat_prop = self.backward_trunk(feat_prop)
    feat_prop = inp[i]
    out.insert(0, feat_prop)
  if last:
    print('calcBackward last', [out[i:i+3] for i in range(len(out) - 3)])
  return [out[i:i+3] for i in range(len(out) - 3)] # only window[0] is used
def calcFlowForward(flowInp, **_):
  out = []
  if flowForward.first:
    out.append(None)
    flowInp = flowInp[1:]
    flowForward.first = 0
  # flowInp = self.get_forward_flow(flowInp) [[x_i, x_i+1], ...]
  out.extend(list(flowInp))
  return out
def calcForward(inp, flowInp, keyframeFeature, backward, **_):
  n = inp.shape[0]
  print('calcForward', inp, flowInp, keyframeFeature, backward)
  feat_prop = inp.new_zeros(1, 1) # batch, channel, height, width
  out = []
  for i in range(n):
    if flowInp[i] != None:
      # feat_prop = self.flow_warp(feat_prop, flowInp[i])
      pass
    if keyframeFeature[i] != None:
      # feat_prop = torch.cat([feat_prop, keyframeFeature[i]], dim=1)
      # feat_prop = self.forward_fusion(feat_prop)
      assert keyframeFeature[i][3] == inp[i]
    feat_prop = inp[i] + feat_prop
    out.append(feat_prop)
    # feat_prop = torch.cat([inp[i], backward[i][0], feat_prop], dim=1)
    # feat_prop = self.forward_trunk(feat_prop)
  return out

startPadding = 2
endPadding = -3
inp = StreamState(name='inp')
inp1 = StreamState(name='inp1')
inp2 = StreamState(name='inp2')
backwardInp = StreamState(offload=True, name='backwardInp')
flowInp = StreamState(2, name='flowInp')
flowForwardInp = StreamState(offload=True, name='flowForwardInp').setPadding(1)
flowBackwardInp = StreamState(name='flowBackwardInp')
isKeyFrame = KeyFrameState(7, 0)
keyframeFeatureInp = StreamState(7, tensor=False, reserve=1, name='keyframeFeatureInp').setPadding(startPadding).setPadding(endPadding)
StreamState.pipe(pp, [inp], [inp1, inp2, flowInp, backwardInp], args=['input fan out'])
StreamState.pipe(pp, [flowInp], [flowForwardInp, flowBackwardInp], args=['flow fan out'])
keyframeFeature = StreamState(tensor=False, name='keyframeFeature')
StreamState.pipe(getKeyframeFeature, [keyframeFeatureInp, isKeyFrame], [keyframeFeature], size=3)
keyframeFeature1 = StreamState(tensor=False, name='keyframeFeature1')
keyframeFeature2 = StreamState(tensor=False, name='keyframeFeature2')
StreamState.pipe(pp, [keyframeFeature], [keyframeFeature1, keyframeFeature2], args=['keyframe feature fan out'])
flowBackward = StreamState(tensor=False, name='flowBackward')
StreamState.pipe(calcFlowBackward, [flowBackwardInp], [flowBackward], size=4)
backward = StreamState(tensor=False, name='backward')
StreamState.pipe(calcBackward, [backwardInp, flowBackward, keyframeFeature1], [backward], size=20)
flowForward = StreamState(tensor=False, name='flowForward')
flowForward.first = 1 # signal alignment for frame 0, 1
StreamState.pipe(calcFlowForward, [flowForwardInp], [flowForward], size=4)
forward = StreamState(name='forward')
StreamState.pipe(calcForward, [inp1, flowForward, keyframeFeature2, backward], [forward])
upsample = StreamState(name='upsample', store=False)
out = StreamState.pipe(pp, [inp2, forward], [upsample], args=['upsample'])
test = torch.arange(32)
print(out.send((None, 2)))
for i, t in enumerate(test):
  if i + startPadding > 2:
    inp.push(t.unsqueeze(0))
  keyframeFeatureInp.push(t.unsqueeze(0))
  print(out.send(None))
print('finish streaming')
while True:
  try:
    print(out.send(True))
  except StopIteration: break
print('finish')