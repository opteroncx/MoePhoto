import os

def getGPU():
    print('reading gpu info')
    nv = os.popen('nvidia-smi -q -d Memory')
    ginfo = nv.readlines()
    free = ginfo[11]
    free = free.split(':')[1].split('MiB')[0].strip('\\n| ')
    free = int(free)
    return free

def getName():
    nv = os.popen('nvidia-smi -L')
    name = nv.readlines()
    print(name)
    return name

if __name__ == '__main__':
    getGPU()