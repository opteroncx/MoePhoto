from sys import platform

class PipeWin():
  def __init__(self, name):
    self.src = pipeT.format('src-', name)
    self.dst = pipeT.format('dst-', name)
    self.psrc = win32pipe.CreateNamedPipe(self.src,
      win32pipe.PIPE_ACCESS_INBOUND | win32file.FILE_FLAG_OVERLAPPED,
      win32pipe.PIPE_TYPE_BYTE,
      1, 0, bufsize, 300, None)
    self.pdst = win32pipe.CreateNamedPipe(self.dst,
      win32pipe.PIPE_ACCESS_OUTBOUND | win32file.FILE_FLAG_OVERLAPPED,
      win32pipe.PIPE_TYPE_BYTE,
      1, 0, 0, 300, None)
    self.open = True
    self._size = 0
    self._info = 0

  def getSrc(self):
    return self.src

  def getDst(self):
    return self.dst

  def _write(self):
    size = self._size
    if not (size and self._info):
      return 0
    err, data = win32file.ReadFile(self.psrc, size)
    print(data[:8], err)
    win32file.WriteFile(self.pdst, data)
    self._size = 0
    return size

  def transmit(self):
    if not self.open:
      return 0
    try:
      _, size, err = win32pipe.PeekNamedPipe(self.psrc, 0)
      if err:
        raise RuntimeError('Read failed on {}.'.format(self.src))
      if size:
        self._size = size
        if not self._info:
          try:
            self._info = win32pipe.GetNamedPipeClientProcessId(self.pdst)
            print(self._info)
          except: pass
        return self._write()
    except Exception as e:
      if hasattr(e, 'winerror') and e.winerror in (109, errno.ESHUTDOWN):  # pylint: disable=E1101
        self.open = False
        return 0
      raise e

  def close(self, last=False):
    if last:
      self.transmit()
    self._write()
    if self._size and not self._info:
      print('waiting for output ', self.dst)
      win32pipe.ConnectNamedPipe(self.pdst, None)
      self._write()
    self.psrc.close()
    self.pdst.close()
    self.open = False

class PipeUnix():
  def __init__(self, name):
    self.path = uploadDir + '/' + name
    os.mkfifo(self.path)  # pylint: disable=E1101

  def getSrc(self):
    return self.path

  def getDst(self):
    return self.path

  def transmit(self): pass

  def close(self):
    os.unlink(self.path)

class DummyPipe():
  def __init__(self, _=None): pass

  def getSrc(self):
    return ''

  def getDst(self):
    return ''

  def transmit(self): pass

  def close(self): pass

if platform[:3] == 'win':
  import win32pipe, win32file
  import errno
  pipeT = r'\\.\pipe\{}{}'
  bufsize = 2 ** 20
  Pipe = lambda name, dummy: DummyPipe() if dummy else PipeWin(name)
else:
  import os
  from config import config
  uploadDir = config.uploadDir  # pylint: disable=E1101
  Pipe = lambda name, dummy: DummyPipe() if dummy else PipeUnix(name)