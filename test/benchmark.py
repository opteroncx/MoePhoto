
# coding: utf-8

# In[1]:


import torch
import sys
sys.path.append('./python')
from progress import initialETA, ops, newOp, slideAverage
from imageProcess import genProcess
from config import config

inTensor = torch.randn((3, 1080, 1920), dtype=config.dtype(), device=config.device())

imgType = dict(bitDepth=8, channel=0, source=0, load=inTensor.nelement())

opt1 = dict(op='SR', model='a', scale=2)
opt2 = dict(op='SR', model='lite', scale=2)

def run(cases, times=1):
    for _, node, _1 in cases:
        node.learn = times
        ops[node.op] = newOp(slideAverage(1 - 1 / times))
    for i in range(times):
        for process, node, _ in cases:
            initialETA(node)
            process(inTensor)
            torch.cuda.empty_cache()

def show(cases):
    for _, node, opt in cases:
        v = opt.copy()
        if 'opt' in v:
            del v['opt']
        print(v, ops[node.op].weight)


# In[2]:


p1, nodes1 = genProcess([opt1], True, imgType)
p2, nodes2 = genProcess([opt2], True, imgType)
cases = [(p1, nodes1[0], opt1), (p2, nodes2[0], opt2)]
run(cases) # warming up


# In[3]:


run(cases, 10)
show(cases)