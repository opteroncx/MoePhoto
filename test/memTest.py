import sys
sys.path.append('./python')
import torch
from procedure import genProcess
from config import config, process

getMemUsed = (lambda i: torch.cuda.memory_stats(i)['reserved_bytes.all.peak']) if config.cuda else lambda *_: process.memory_info()[0]
step = dict(op='dehaze', model='moire_obj')
load = 3 << 20
p, _ = genProcess([step], True, dict(bitDepth=8, channel=0, source=0, load=load))
t = torch.randn((3, 1440, 900), dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
m = getMemUsed(config.device())
print(config.dtype(), m)
sys.stdin.readline()
if config.cuda:
  getMemUsed(config.device())
p(t)
m = getMemUsed(config.device()) if config.cuda else (getMemUsed() - m)
print(m, m / load)
sys.stdin.readline()