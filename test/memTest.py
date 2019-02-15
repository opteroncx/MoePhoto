import sys
sys.path.append('./python')
import torch
from imageProcess import genProcess
from config import config

p, _ = genProcess([dict(op='SR', scale=2, model='lite')], True, dict(bitDepth=8, channel=0, source=0, load=1 << 16))
t = torch.randn((1, 510, 510), dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
p(t)