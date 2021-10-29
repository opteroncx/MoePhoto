import sys
import os
import torch
from torch import nn
sys.path.append(os.path.abspath('../python'))
sys.path.append('../site-packages/nvidia-ml-py')
from PIL import Image
import numpy as np
from config import config
config.dir = '..'
config.initialize()
from imageProcess import readFile, initModel, toFloat, toOutput, ensemble, writeFile, Option
show = lambda im: Image.fromarray(toOutput(8)(im).transpose(1, 2, 0))
toTorch = lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).to(dtype=config.dtype(), device=config.device()) / 256
def context():pass
readFile = readFile(context=context)
readPic = lambda path: toTorch(readFile(path))
modules = None

import re
from functools import reduce
getRoot = lambda w, r: tuple(filter(lambda s: s.startswith(r), w.keys()))
replaces = lambda w, r, p, s: list((t, t.replace(p, s)) for t in getRoot(w, r + p))
inserts = lambda w, r, p, s: list((t, t.replace(r + p, r + s + p)) for t in getRoot(w, r + p))
getMatch = lambda o: o.group(1 if len(o.groups()) else 0)
find = lambda names, r: set(getMatch(o) for o in filter(None, map(re.compile(r).match, names)))
findRs = lambda w, rs: reduce(set.union, (find(w.keys(), r) for r in rs))
def changeName(w, old, new):
  if not old in w: return
  if new:
    w[new] = w[old]
    print('rename {} to {}'.format(old, new))
  del w[old]
def changeNames(w, names):
  values = dict((new, w[old]) for old, new in names)
  for old, _ in names:
    if old in w:
      del w[old]
  w.update(values)
pf = lambda f, w, r: lambda p, s: [] if p is None else f(w, r, p, s)
getNames = lambda w: lambda rst: reduce(lambda a, r: a + pf(replaces, w, r)(rst[1], rst[2]) + pf(inserts, w, r)(rst[3], rst[4]), findRs(w, rst[0]), [])
getSubNames = lambda w: lambda rst: [(k, re.sub(rst[0], rst[1], k)) for k in w.keys() if re.match(rst[0], k)]
cc = lambda w, f=getNames: lambda rsts: tuple(changeNames(w, f(w)(rst)) for rst in rsts)
removeRoot = lambda w, r: tuple(changeName(w, old, None) for old in getRoot(w, r))
fm1 = lambda i, s0, s1, s2: (s0.format(i + 1), s1.format(i + 1), s2.format(i))
reT = lambda t: ((t[0],), t[1], t[2], None, None)
# {'{name}.*' => '{name}': {'*'} for name in rules}
namespaced = lambda w, rules: dict(
  (name, dict((key.removeprefix('{}.'.format(name)), w[key]) for key in getRoot(w, name))) for name in rules
)

def renameByRules(models, rsts='', subs='', rules=0):
  for mC, kwargs, root, *paths in models:
    m = mC(**kwargs) if mC else None
    for modelPath in paths:
      print(modelPath)
      weights = torch.load(modelPath, map_location='cpu')
      if root:
        weights = weights[root]
      cc(weights)(rsts)
      cc(weights, getSubNames)(subs)
      if rules:
        weights = namespaced(weights, rules)
      if m:
        print(m.load_state_dict(weights))
        weights = m.state_dict()
      torch.save(weights, modelPath + '.new', pickle_protocol=4)
  return m if m else weights

def pp(m, l=0):
  t = '\t' * l
  for key in m:
    v = m[key]
    tl = t + str(key)
    if type(v) == torch.Tensor:
      print(tl, v.shape)
    elif type(v) in {bool, int, float, str, list, tuple, set}:
      print(tl, v)
    else:
      print(tl + ':')
      pp(v, l + 1)