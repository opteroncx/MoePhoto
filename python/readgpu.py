import sys
import torch

ss = lambda f: lambda *args: str(f(*args), encoding='ascii')
def init():
  global devices, getGPUName, pynvml
  sys.path.append('site-packages/nvidia-ml-py')
  import pynvml  # pylint: disable=E0401
  sys.path.pop()
  pynvml.nvmlInit()
  devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
  getGPUName = ss(pynvml.nvmlDeviceGetName)

def getGPUProperty(i):
  prop = torch.cuda.get_device_properties(i)
  return {
    'name': prop.name,
    'capability': prop.major + prop.minor / 10,
    'total_memory': prop.total_memory // 2**20,
    'processor_count': prop.multi_processor_count
  }

def uninstall():
  if 'readgpu' in sys.modules:
    del sys.modules['readgpu']

getFreeMem = lambda device: pynvml.nvmlDeviceGetMemoryInfo(device).free  # pylint: disable=E0602
getGPU = lambda: list(map(getFreeMem, devices))
getName = lambda: list(map(getGPUName, devices))
getPythonVersion = lambda: '.'.join(map(str, sys.version_info[:3]))
getTorchVersion = lambda: torch.__version__
getGPUProperties = lambda: [getGPUProperty(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

if __name__ == '__main__':
  init()
  print(getName(), getGPU())
  print(getPythonVersion())
  print(getTorchVersion())
  for prop in getGPUProperties():
    print(prop)