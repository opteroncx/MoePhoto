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
getMatch = lambda o: o.group(1) if len(o.groups()) else o.group(0)
def find(names, r):
    r = re.compile(r)
    return set(map(getMatch, filter(None, map(r.match, names))))
findRs = lambda w, rs: reduce(set.union, (find(tuple(w.keys()), r) for r in rs))
def changeName(w, old, new):
    if new:
        w[new] = w[old]
    del w[old]
def changeNames(w, names):
    for old, new in names:
        changeName(w, old, new)
pf = lambda f, w, r: lambda p, s: [] if p is None else f(w, r, p, s)
getNames = lambda w: lambda rst: reduce(lambda a, r: a + pf(replaces, w, r)(rst[1], rst[2]) + pf(inserts, w, r)(rst[3], rst[4]), findRs(w, rst[0]), [])
cc = lambda w: lambda rsts: tuple(changeNames(w, getNames(w)(rst)) for rst in rsts)
removeRoot = lambda w, r: tuple(changeName(w, old, None) for old in getRoot(w, r))
fm1 = lambda i, s0, s1, s2: (s0.format(i + 1), s1.format(i + 1), s2.format(i))
reT = lambda t: ((t[0],), t[1], t[2], None, None)