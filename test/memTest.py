import sys
sys.path.append('./python')
import torch
from config import config, process
from procedure import genProcess

getMemUsed = (lambda i: torch.cuda.memory_stats(i)['reserved_bytes.all.peak']) if config.cuda else lambda *_: process.memory_info()[1]
step = dict(op='SR', model='gan', scale=2)
t = torch.randn((3, 512, 512), dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
load = t.nelement()
p, _ = genProcess([step], True, dict(bitDepth=8, channel=0, source=0, load=load))
m = getMemUsed(config.device())
print(config.dtype(), m)
sys.stdin.readline()
if config.cuda:
  getMemUsed(config.device())
p(t)
m = getMemUsed(config.device()) if config.cuda else (getMemUsed() - m)
print(m, m / load)
sys.stdin.readline()