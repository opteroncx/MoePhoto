import sys
sys.path.append('./python')
from time import perf_counter
import torch
from torch.profiler import profile, schedule, ProfilerActivity
from config import config
from imageProcess import doCrop

from videoSR import getOpt
modelConfig = None
shape = (1, 2, 3, 1088, 1920)

p = getOpt(modelConfig).spynet#.modelCached
#p.outShape = (1, 64, 1088, 1920)

getMemUsed = lambda i: torch.cuda.memory_stats(i)['reserved_bytes.all.peak']
t = torch.randn(shape, dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
load = shape[-1] * shape[-2] * shape[0]
m = getMemUsed(config.device()) if config.cuda else None
print(config.dtype(), config.device(), m)
if config.cuda:
  p(t)
  #doCrop(p, t)
  getMemUsed(config.device())
  start = perf_counter()
  p(t)
  #doCrop(p, t).mean().cpu()
  print('time elpased: {}'.format(perf_counter() - start))
  m = getMemUsed(config.device())
else:
  schedule1 = schedule(
    wait=1,
    warmup=1,
    active=1)
  with profile(
    activities=[ProfilerActivity.CPU],
    schedule=schedule1, profile_memory=True) as pro:
    for _ in range(3):
      p(t)
      pro.step()
    avg = pro.key_averages()
    avg.sort(key=lambda o: o.cpu_memory_usage, reverse=True)
    m = avg[0].cpu_memory_usage
print(m, m / load, load)