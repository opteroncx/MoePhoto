import numpy as np
from torch.nn import ReflectionPad2d
from python.imageProcess import identity, ceilBy
minSize = 28
alignF = {
  1: identity,
  8: ceilBy(8),
  32: ceilBy(32)
}
def getAnchors(s, ns, l, pad, af, sc):
  a = af(s)
  n = l - 2 * pad
  step = int(np.ceil(ns / n))
  start = np.arange(step, dtype=np.int) * n + pad
  start[0] = 0
  end = start + l
  if step > 1:
    start[-1] = s - l
    end[-1] = s
    clip = [(end[-2] - s - pad) * sc, l * sc]
  else:
    end[-1] = a
    clip = [0, s * sc]
  # clip = [0:l - pad, pad:l - pad, ..., end[-2] - s - pad:l]
  return start, end, clip, step

def prepare(shape, ramCoef, ram, pad=0, align=8, sc=1):
  c, h, w = shape
  n = ram * ramCoef / c
  af = alignF[align]
  s = af(minSize + pad * 2)
  outh, outw = h * sc, w * sc
  if n < s * s:
    raise MemoryError('Free memory space is not enough.')
  ph, pw = max(1, h - pad * 2), max(1, w - pad * 2)
  ns = np.arange(s / align, int(n / (align * s)) + 1, dtype=np.int)
  ms = (n / (align * align) / ns).astype(int)
  ns, ms = ns * align, ms * align
  ds = np.ceil(ph / (ns - 2 * pad)) * np.ceil(pw / (ms - 2 * pad)) # minimize number of clips
  ind = np.argwhere(ds == ds.min()).squeeze(1)
  mina = ind[np.abs(ind - len(ds) / 2).argmin()] # pick the size with ratio of width and height closer to 1
  ah, aw = af(h), af(w)
  ih, iw = min(ah, ns[mina]), min(aw, ms[mina])
  startH, endH, clipH, stepH = getAnchors(h, ph, ih, pad, af, sc)
  startW, endW, clipW, stepW = getAnchors(w, pw, iw, pad, af, sc)
  padImage = identity if (stepH > 1) and (stepW > 1) else ReflectionPad2d((0, (aw - w) * (stepW < 2), 0, (ah - h) * (stepH < 2)))
  eh, ew = (ih - pad) * sc, (iw - pad) * sc
  padSc, ihSc, iwSc = pad * sc, ih * sc, iw * sc
  def iterClip():
    for i in range(stepH):
      top, bottom = startH[i], endH[i]
      topT, bottomT = clipH if i == stepH - 1 else ((0, ihSc) if i == 0 else (padSc, eh))
      for j in range(stepW):
        left, right = startW[j], endW[j]
        leftT, rightT = clipW if j == stepW - 1 else ((0, iwSc) if j == 0 else (padSc, ew))
        yield (top, bottom, left, right, topT, bottomT, leftT, rightT)
  return iterClip, padImage, (c, outh, outw)