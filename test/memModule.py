import sys
sys.path.append('./python')
from time import perf_counter
import torch
from torch.profiler import profile, schedule, ProfilerActivity
from config import config
# from imageProcess import doCrop
shape = (1, 2, 3, 1088, 1920)
getMemUsedBy = lambda stats, key: stats['active_bytes.all.{}'.format(key)] + stats['inactive_split_bytes.all.{}'.format(key)]
getMemUsed = lambda key='current': getMemUsedBy(torch.cuda.memory_stats(config.device()), key)
t = torch.randn(shape, dtype=config.dtype(), device=config.device()) # pylint: disable=E1101

from IFRNet import getOpt, Channels
modelConfig = {'model': 'L', 'sf': 1}
chsOut = [cOut[0] if type(cOut) is tuple else cOut for cOut in Channels[modelConfig['model']]]

p = getOpt(modelConfig).decoder
#p.outShape = (1, 64, 1088, 1920)

p.setSize(shape[3], shape[4], t)
args = [[(torch.tensor([.5]).to(t), 0, 0)]]
t = [torch.randn([shape[0], shape[1], cOut, shape[3] >> (4 - i), shape[4] >> (4 - i)], dtype=t.dtype, device=t.device) for i, cOut in enumerate(reversed(chsOut))]
load = shape[-1] * shape[-2] * shape[0]
m = getMemUsed() if config.cuda else None
print(config.dtype(), config.device(), m)
if config.cuda:
  p(t, *args)
  torch.cuda.synchronize(config.device())
  #doCrop(p, t)
  mPre = getMemUsed()
  start = perf_counter()
  p(t, *args)
  #doCrop(p, t).mean().cpu()
  torch.cuda.synchronize(config.device())
  print('time elpased: {}'.format(perf_counter() - start))
  m = getMemUsed('peak')
else:
  mPre = 0
  schedule1 = schedule(
    wait=1,
    warmup=1,
    active=1)
  with profile(
    activities=[ProfilerActivity.CPU],
    schedule=schedule1, profile_memory=True) as pro:
    for _ in range(3):
      p(t, *args)
      pro.step()
    avg = pro.key_averages()
    avg.sort(key=lambda o: o.cpu_memory_usage, reverse=True)
    m = avg[0].cpu_memory_usage
m -= mPre
print(m, m / load, load)