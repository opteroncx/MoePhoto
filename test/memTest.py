import sys
sys.path.append('./python')
import torch
from imageProcess import genProcess
from config import config

step = dict(op='SR', scale=4, model='lite')
p, _ = genProcess([step], True, dict(bitDepth=8, channel=0, source=0, load=1 << 18))
t = torch.randn((1, 510, 510), dtype=config.dtype(), device=config.device()) # pylint: disable=E1101
print(config.dtype())
p(t)