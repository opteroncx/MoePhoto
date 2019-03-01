import time
from gevent.queue import Queue
from gevent import spawn_later

def runPeriod(func, period):
  flag = True
  def f():
    if flag:
      spawn_later(period, f)
      return func()
  spawn_later(period, f)
  def stop():
    nonlocal flag
    flag = False
  return stop

Null = lambda *_: None

class Cache():
  def __init__(self, size, lifetime=3600, onExtinct=Null):
    self.lifetime = lifetime
    self.queue = Queue(size)
    self.stop = runPeriod(self.clean, lifetime)
    self.out = onExtinct

  def put(self, item):
    t = (time.time(), item)
    while True:
      try:
        self.queue.put_nowait(t)
      except:
        self.clean(True)
      else:
        break
    return self.queue.qsize()

  def clean(self, force=False):
    if not hasattr(self, 'queue'):
      return 0
    old = time.time() - self.lifetime
    count = 0
    while not self.queue.empty():
      t = self.queue.peek_nowait()[0]
      if t < old:
        self.out(self.queue.get_nowait()[1])
        count += 1
      else:
        break
    if force and self.queue.qsize() == self.queue.maxsize:
      self.out(self.queue.get_nowait()[1])
    return count

  def destroy(self):
    self.stop()
    del self.queue
    del self