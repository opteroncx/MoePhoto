import sys
sys.path.append('./python')
import torch
from torch.profiler import profile, schedule, ProfilerActivity
from config import config
from procedure import genProcess

getMemUsed = lambda i: torch.cuda.memory_stats(i)['reserved_bytes.all.peak']
step = dict(op='dehaze', model='NAFNet_deblur_32', scale=1)
t = torch.randn((3, 1024, 1024), dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
load = t.nelement()
p, _ = genProcess([step], True, dict(bitDepth=8, channel=0, source=0, load=load))
m = getMemUsed(config.device()) if config.cuda else None
print(config.dtype(), config.device(), m, load)
sys.stdin.readline()
if config.cuda:
  p(t)
  getMemUsed(config.device())
  p(t)
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
print(m, m / load)
sys.stdin.readline()