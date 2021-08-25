import sys
import os
import torch
from torch import nn
sys.path.append(os.path.abspath('../python'))
sys.path.append('../site-packages/nvidia-ml-py')
from PIL import Image
import numpy as np
from imageProcess import readFile, initModel, toFloat, toOutput, ensemble, writeFile, Option
from config import config
config.dir = '..'
config.initialize()
show = lambda im: Image.fromarray(toOutput(8)(im).transpose(1, 2, 0))
toTorch = lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).to(dtype=config.dtype(), device=config.device()) / 256
def context():pass
readFile = readFile(context=context)
readPic = lambda path: toTorch(readFile(path))

import re
from functools import reduce
getRoot = lambda w, r: tuple(filter(lambda s: s.startswith(r), w.keys()))
replaces = lambda w, r, p, s: list((t, t.replace(p, s)) for t in getRoot(w, r + p))
inserts = lambda w, r, p, s: list((t, t.replace(r + p, r + s + p)) for t in getRoot(w, r + p))
getMatch = lambda o: o.group(1 if len(o.groups()) else 0)
find = lambda names, r: set(getMatch(o) for o in filter(None, map(re.compile(r).match, names)))
findRs = lambda w, rs: reduce(set.union, (find(w.keys(), r) for r in rs))
def changeName(w, old, new):
  if new:
    w[new] = w[old]
  del w[old]
changeNames = lambda w, names: [changeName(w, old, new) for old, new in names]
pf = lambda f, w, r: lambda p, s: [] if p is None else f(w, r, p, s)
getNames = lambda w: lambda rst: reduce(lambda a, r: a + pf(replaces, w, r)(rst[1], rst[2]) + pf(inserts, w, r)(rst[3], rst[4]), findRs(w, rst[0]), [])
getSubNames = lambda w: lambda rst: [(k, re.sub(rst[0], rst[1], k)) for k in w.keys() if re.match(rst[0], k)]
cc = lambda w, f=getNames: lambda rsts: tuple(changeNames(w, f(w)(rst)) for rst in rsts)
removeRoot = lambda w, r: tuple(changeName(w, old, None) for old in getRoot(w, r))
fm1 = lambda i, s0, s1, s2: (s0.format(i + 1), s1.format(i + 1), s2.format(i))
reT = lambda t: ((t[0],), t[1], t[2], None, None)

def renameByRules(models, rsts='', subs=''):
  for mC, kwargs, root, *paths in models:
    m = mC(**kwargs)
    for modelPath in paths:
      print(modelPath)
      weights = torch.load(modelPath, map_location='cpu')
      if root:
        weights = weights[root]
      cc(weights)(rsts)
      cc(weights, getSubNames)(subs)
      m.load_state_dict(weights)
      torch.save(m.state_dict(), modelPath + '.new', pickle_protocol=4)
  return m

def pp(m, l=0):
  for key in m:
    if type(m[key]) == torch.Tensor:
      print(('\t' * l) + key, m[key].shape)
    else:
      print(('\t' * l) + key + ':')
      pp(m[key], l + 1)