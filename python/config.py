# pylint: disable=E1101
import time
import os
import json
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
  def __init__(self):
    self.deviceId = 0
    try:
      setConfig(self.__dict__, VERSION)
    except Exception as e:
      log.warning(e)
    self.cuda &= Config.cudaAvailable()
    try:
      readgpu.init()
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
      free_ram = readgpu.getGPU()[self.deviceId]
    else:
      mem = psutil.virtual_memory()
      free_ram = mem.free
    self.freeRam = free_ram
    return free_ram

  def calcFreeMem(self):
    if self.cuda:
      memUsed = torch.cuda.memory_allocated(self.deviceId) * 4
      if self.maxGraphicMemoryUsage > 0:
        free = min(self.freeRam, self.maxGraphicMemoryUsage * 2**20) - memUsed
      else:
        free = self.freeRam - memUsed
    else:
      memUsed = process.memory_info()[0]
      self.freeRam = psutil.virtual_memory().free
      if self.maxMemoryUsage > 0:
        free = min(self.freeRam - 2**28, self.maxMemoryUsage * 2**20 - memUsed)
      else:
        free = self.freeRam - 2**28
    return free

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
      self.freeRam = freeMems[self.deviceId]
      gram = [freeMem // 2**20 for freeMem in freeMems]
    except Exception as e:
      print(e)
      gram = []
    return gram

Config.cudaAvailable = lambda *args:torch.cuda.is_available()

config = Config()

if __name__ == '__main__':
  config.getFreeMem()
  print(config.calcFreeMem())