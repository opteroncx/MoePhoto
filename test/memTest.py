import sys
sys.path.append('./python')
import torch
from procedure import genProcess
from config import config, process

getMemUsed = torch.cuda.max_memory_cached if config.cuda else lambda: process.memory_info()[0]
step = dict(op='dehaze', model='sun')
load = 3 << 20
p, _ = genProcess([step], True, dict(bitDepth=8, channel=0, source=0, load=load))
t = torch.randn((3, 1024, 1024), dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
m = getMemUsed()
print(config.dtype(), m)
sys.stdin.readline()
if config.cuda:
  torch.cuda.reset_max_memory_cached()
p(t)
m = getMemUsed() if config.cuda else (getMemUsed() - m)
print(m, m / load)
sys.stdin.readline()