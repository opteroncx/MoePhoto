from turbojpeg import TurboJPEG, TJSAMP_420, TJFLAG_FASTDCT
from PIL import Image
from time import perf_counter
import numpy as np
from mmap import mmap
mm = mmap(-1, 2 ** 24, tagname='SharedMemory')

def timing(n, f, initial):
  t = 0
  for i in range(n):
    initial()
    start = perf_counter()
    f(i)
    t += perf_counter() - start
  return t

initial = lambda: mm.seek(0)
jpeg = TurboJPEG("C:/Users/lotress/MoePhoto/test/libturbojpeg.dll")
filePath = r'C:\Users\lotress\Documents\福州轨道交通线路图（2050+）@chinho.jpg'
imgFile = open(filePath, 'rb')
imgBuf = imgFile.read()
imgFile.seek(0)
img1 = jpeg.decode(imgBuf)
img2 = Image.fromarray(np.array(Image.open(imgFile)))
imgFile.close()
f1 = lambda kwargs: lambda _: mm.write(jpeg.encode(img1, **kwargs))
f2 = lambda _: img2.save(mm, 'jpeg')
print('Timing JPEG encoding by libjpeg-turbo: ', timing(1, f1({'jpeg_subsample': TJSAMP_420}), initial))
print('Timing JPEG encoding by Pillow: ', timing(1, f2, initial))