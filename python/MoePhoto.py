import sys
from mmap import mmap
from defaultConfig import defaultConfig

sharedMemSize = defaultConfig['sharedMemSize'][0]
mm = mmap(-1, sharedMemSize, tagname='SharedMemory') if sys.platform[:3] == 'win' else mmap(-1, sharedMemSize)

def lock(duration):
  from gevent.event import Event
  flag = Event()
  node = begin(Node({}, 1, duration, 0))
  node.reset().trace(0)
  while duration > 0 and not context.stopFlag.is_set():
    duration -= 1
    flag.wait(1)
    flag.clear()
    node.trace()
  return duration

def imageEnhance(size, *args, name=None, trace=True):
  process, nodes = genProcess(stepFile + list(args))
  return begin(imNode, nodes, True if trace else -1).bindFunc(process)(size, name=name)

if __name__ == '__main__':
  import multiprocessing as mp
  from server import runserver
  from worker import worker
  from defaultConfig import defaultConfig
  taskInReceiver, taskInSender = mp.Pipe(False)
  taskOutReceiver, taskOutSender = mp.Pipe(False)
  noter, notifier = mp.Pipe(False)
  stopEvent = mp.Event()
  mp.Process(target=worker, args=(taskInReceiver, taskOutSender, notifier, stopEvent), daemon=True).start()
  run = runserver(taskInSender, taskOutReceiver, noter, stopEvent, notifier, mm)
  host = '127.0.0.1'
  port = defaultConfig['port'][0]
  if len(sys.argv) > 1:
    if '-g' in sys.argv:
      host = ''
    run(host, port)
  else:
    from webbrowser import open as startBrowser
    from gevent import spawn_later
    spawn_later(1, startBrowser, 'http://127.0.0.1:{}'.format(port))
    run(host, port)
else:
  from progress import Node
  from worker import begin, setup, context, enhance
  from imageProcess import genProcess
  from video import SR_vid
  from config import config
  stepFile = [{'op': 'file'}]
  imNode = Node({'op': 'image'}, learn=0)
  setup(mm, {
    'lock': lock,
    'image_enhance': enhance(imageEnhance),
    'video_enhance': enhance(SR_vid),
    'ednoise_enhance': enhance(imageEnhance),
    'image_dehaze': enhance(imageEnhance),
    'systemInfo': enhance(config.system)
  })