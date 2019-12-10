import sys

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
      1, bufsize, 0, 300, None)

  def getSrc(self):
    return self.src

  def getDst(self):
    return self.dst

  def transmit(self):
    try:
      _, size, err = win32pipe.PeekNamedPipe(self.psrc, 0)
      if err:
        raise RuntimeError('Read failed on {}.'.format(self.src))
      if size:
        err, data = win32file.ReadFile(self.psrc, size)
        win32file.WriteFile(self.pdst, data)
    except Exception as e:
      if hasattr(e, 'winerror') and e.winerror in (109, errno.ESHUTDOWN):  # pylint: disable=E1101
        self.close()
        return 0
      raise e
    return size

  def close(self):
    self.psrc.close()
    self.pdst.close()

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

isWindows = sys.platform[:3] == 'win'
if isWindows:
  import win32pipe, win32file
  import errno
  pipeT = r'\\.\pipe\{}{}'
  bufsize = 2 ** 20
  Pipe = PipeWin
else:
  import os
  from config import config
  uploadDir = config.uploadDir  # pylint: disable=E1101
  Pipe = PipeUnix