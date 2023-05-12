# pylint: disable=E1101
import time
import os
import logging
import psutil
import torch
import readgpu
from userConfig import setConfig, VERSION
process = psutil.Process(os.getpid())
log = logging.getLogger('Moe')

def transform(self):
  def f(key):
    v = self.__dict__[key]
    if v == 'auto':
      return 0
    else:
      return v
  return f

class Config():
  def __init__(self, dir='.'):
    self.deviceId = 0
    self.dir = dir
    self.initialize()

  def initialize(self):
    try:
      setConfig(self.__dict__, VERSION, dir=self.dir)
    except Exception as e:
      log.warning(e)
    self.cuda &= Config.cudaAvailable()
    try:
      if self.cuda:
        readgpu.init()
        print('{} bytes of graphical memory available.'.format(self.getFreeMem()))
    except Exception as e:
      log.warning(e)
    if self.cuda and self.deviceId >= torch.cuda.device_count():
      log.warning(RuntimeWarning('GPU #{} not available, using GPU #0 instead'.format(self.deviceId)))
      self.deviceId = 0

  def getConfig(self):
    return tuple(map(transform(self), ('crop_sr', 'crop_dn', 'crop_dns')))

  def getPath(self, **kwargs):
    kwargs['timestamp'] = int(time.time())
    d = dict((key, kwargs[key]) for key in kwargs if key in self.videoName)
    return self.videoName.format(**d)

  def getFreeMem(self, emptyCache=False):
    if self.cuda:
      if emptyCache:
        torch.cuda.empty_cache()
      free_ram = readgpu.getGPU()[self.deviceId] - 2**28
    else:
      mem = psutil.virtual_memory()
      free_ram = mem.free - 2**28
    return free_ram

  def calcFreeMem(self):
    freeRam = self.getFreeMem()
    if self.cuda:
      freeRam = freeRam * .8 + torch.cuda.memory_reserved() * .4
      if self.maxGraphicMemoryUsage > 0:
        memUsed = torch.cuda.memory_allocated()
        freeRam = min(freeRam, self.maxGraphicMemoryUsage * 2**20 - memUsed)
    elif self.maxMemoryUsage > 0:
      memUsed = process.memory_info()[0]
      freeRam = min(freeRam, self.maxMemoryUsage * 2**20 - memUsed)
    return int(freeRam)

  def dtype(self):
    return torch.half if self.cuda and self.fp16 else torch.float

  def device(self):
    return torch.device('cuda:{}'.format(self.deviceId) if self.cuda else 'cpu')

  def getRunType(self):
    if self.cuda:
      return 2 if self.fp16 else 1
    else:
      return 0

  def system(self):
    if not Config.cudaAvailable():
      return []
    try:
      freeMems = readgpu.getGPU()
      gram = [freeMem // 2**20 for freeMem in freeMems]
    except Exception as e:
      print(e)
      gram = []
    return gram

Config.cudaAvailable = lambda *_:torch.cuda.is_available()

config = Config()

if __name__ == '__main__':
  config.getFreeMem()
  print(config.calcFreeMem())