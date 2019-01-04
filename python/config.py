# MoePhoto运行配置文件
# pylint: disable=E1101
import time
import os
from warnings import warn
import psutil
import torch
import readgpu
defaultConfig = {
  'crop_sr': ('auto', '放大模式\n示例-分块大小320像素：\ncrop_sr: 320'),
  'crop_dn': ('auto', '普通降噪'),
  'crop_dns': ('auto', '风格化、强力降噪'),
  'videoName': ('out_{timestamp}.mkv', '输出视频文件名'),
  'maxMemoryUsage': (0, '最大使用的内存MB'),
  'maxGraphicMemoryUsage': (0, '最大使用的显存MB'),
  'cuda': (True, '使用CUDA'),
  'fp16': (True, '使用半精度浮点数'),
  'deviceId': (0, '使用的GPU序号'),
  'defaultCodec': ('libx264 -pix_fmt yuv420p', '默认视频输出编码选项'),
  'ensembleSR': (0, '放大时自集成的扩增倍数， 0-7')
}
process = psutil.Process(os.getpid())

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
    for key in defaultConfig:
      self.__dict__[key] = defaultConfig[key][0]
    self.cuda &= Config.cudaAvailable()
    try:
      readgpu.init()
    except: pass
    if self.cuda and self.deviceId >= torch.cuda.device_count():
      self.deviceId = 0
      warn(RuntimeWarning('GPU #{} not available, using GPU #0 instead'.format(self.deviceId)))

  def getConfig(self):
    return tuple(map(transform(self), ('crop_sr', 'crop_dn', 'crop_dns')))

  def getPath(self):
    return self.videoName.format(timestamp=int(time.time()))

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
        free = self.freeRam
    return free

  def dtype(self):
    dtype = torch.half if self.cuda and self.fp16 else torch.float
    return dtype

  def device(self):
    return torch.device('cuda:{}'.format(self.deviceId) if self.cuda else 'cpu')

  def getRunType(self):
    if self.cuda:
      if self.fp16:
        return 2
      else:
        return 1
    else:
      return 0

  def system(self):
    mem = psutil.virtual_memory()
    mem_total = int(mem.total/1024**2)
    mem_free = int(mem.free/1024**2)
    cpu_count_phy = psutil.cpu_count(logical=False)
    cpu_count_log = psutil.cpu_count(logical=True)
    deviceId = self.deviceId
    try:
      gname = readgpu.getName()[deviceId]
      gram = int(readgpu.getGPU()[deviceId] / 2**20)
      major, minor = torch.cuda.get_device_capability(deviceId)
      ginfo = [gname, gram, major + minor / 10]
      self.ginfo = ginfo
    except Exception as e:
      gerror = '没有检测到NVIDIA的显卡，系统将采用CPU模式'
      ginfo = [gerror, 'N/A', 'N/A']
      print(e)
    return mem_total, mem_free, cpu_count_log, cpu_count_phy, ginfo

Config.cudaAvailable = lambda *args:torch.cuda.is_available()

config = Config()