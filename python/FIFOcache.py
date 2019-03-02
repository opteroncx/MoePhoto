from collections import deque

Null = lambda *_: None

class Cache():
  def __init__(self, size, default=None, onExtinct=Null):
    self.cache = {}
    self.size = size
    self.queue = deque()
    self.default = default
    self.extinct = onExtinct

  def put(self, key, item):
    if len(self.queue) == self.size:
      while len(self.queue):
        oldKey = self.queue.popleft()
        if oldKey in self.cache:
          oldItem = self.cache[oldKey]
          del self.cache[oldKey]
          self.extinct(oldKey, oldItem)
          break
    self.cache[key] = item
    self.queue.append(key)

  def pop(self, key):
    if key in self.cache:
      res = self.cache[key]
      del self.cache[key]
      return res
    else:
      return self.default

  def peek(self, key):
    return key in self.cache