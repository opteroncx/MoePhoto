# MoePhoto运行配置文件
# pylint: disable=E1101
import time
import psutil
import torch
import readgpu
defaultConfig = {
  # 放大模式
  # 示例-分块大小320像素：
  # crop_sr: 320
  'crop_sr': 'auto',
  # 普通降噪
  'crop_dn': 'auto',
  # 风格化、强力降噪
  'crop_dns': 'auto',
  'video_out': 'download',
  'videoName': 'out_{timestamp}.mkv'
}

class Config():
  def __init__(self):
    for key in defaultConfig:
      v = defaultConfig[key]
      if v == 'auto':
        v = 0
      self.__dict__[key] = v

  def getConfig(self):
    return self.crop_sr, self.crop_dn, self.crop_dns

  def getPath(self):
    return self.video_out, self.videoName.format(timestamp=int(time.time()))

Config.cudaAvailable = lambda *args:torch.cuda.is_available()

def getFreeMem(emptyCache=False):
  if Config.cudaAvailable():
    runType = 0
    if emptyCache:
      torch.cuda.empty_cache()
    free_ram = readgpu.getGPU()
  else:
    runType = 1
    mem = psutil.virtual_memory()
    free_ram = mem.free
    free_ram = free_ram/1024**2
    # 预留内存防止系统卡死
    free_ram -= 300
  return runType, free_ram
Config.getFreeMem = getFreeMem

config = Config()