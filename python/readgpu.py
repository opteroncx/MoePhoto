import sys
sys.path.append('site-packages/nvidia-ml-py')
import pynvml  # pylint: disable=E0401
sys.path.pop()
pynvml.nvmlInit()

ss = lambda f: lambda *args: str(f(*args), encoding='ascii')
devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
getGPUName = ss(pynvml.nvmlDeviceGetName)
getFreeMem = lambda device: pynvml.nvmlDeviceGetMemoryInfo(device).free
getGPU = lambda: tuple(map(getFreeMem, devices))
getName = lambda: tuple(map(getGPUName, devices))

if __name__ == '__main__':
  print(getName(), getGPU())