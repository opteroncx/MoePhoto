import sys
sys.path.append('./python')
import torch
from imageProcess import genProcess
from config import config

p, _ = genProcess([dict(op='slomo', sf=2, model='lite')], True, dict(bitDepth=8, channel=0, source=0, load=1920 * 1024))
t = torch.randn((2, 3, 1920, 1024), dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
p(t[0])
p(t[1])
