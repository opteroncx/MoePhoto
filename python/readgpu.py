import os

def getGPU():
  print('reading gpu info')
  nv = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits')
  ginfo = nv.readlines()
  return int(ginfo[0])

def getName():
  nv = os.popen('nvidia-smi -L')
  name = nv.readlines()
  print(name)
  return name

if __name__ == '__main__':
  print(getName(), getGPU())