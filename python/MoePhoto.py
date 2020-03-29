import sys
import multiprocessing as mp
from defaultConfig import defaultConfig
sharedMemSize = defaultConfig['sharedMemSize'][0]
isWindows = sys.platform[:3] == 'win'
mmName = 'SharedMemory'

def getMM(size, create=True):
  if isWindows:
    from mmap import mmap
    return mmap(-1, size, tagname=mmName)
  else: # requires Python >= 3.8
    from multiprocessing.shared_memory import SharedMemory
    shm = SharedMemory(mmName, create, size)
    return shm

if isWindows:
  from subprocess import Popen
  Popen(['chcp', '65001'], shell=True).wait()

def main():
  from progress import Node
  from worker import begin, context, enhance
  from procedure import genProcess
  from video import SR_vid
  from config import config
  stepFile = [{'op': 'file'}]
  imNode = Node({'op': 'image'}, learn=0)

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

  def imageEnhance(size, *args):
    outputOpt = args[-1]
    name = outputOpt['file'] if 'file' in outputOpt else None
    if not ('op' in outputOpt and outputOpt['op'] == 'output'):
      outputOpt = {}
    bench = outputOpt.get('diagnose', {}).get('bench', False)
    trace = outputOpt.get('trace', False) or bench
    process, nodes = genProcess(stepFile + list(args))
    return begin(imNode, nodes, trace, bench).bindFunc(process)(size, name=name)

  mm = getMM(sharedMemSize, False)

  return mm, {
    'lockInterface': lock,
    'image_enhance': enhance(imageEnhance, verbose=False),
    'batch': enhance(imageEnhance, verbose=False),
    'video_enhance': enhance(SR_vid),
    'systemInfo': enhance(config.system)
  }

if __name__ == '__main__':
  mp.set_start_method('spawn')
  from worker import worker
  taskInReceiver, taskInSender = mp.Pipe(False)
  taskOutReceiver, taskOutSender = mp.Pipe(False)
  noter, notifier = mp.Pipe(False)
  stopEvent = mp.Event()
  mp.Process(target=worker, args=(main, taskInReceiver, taskOutSender, notifier, stopEvent, isWindows), daemon=True).start()
  from server import runserver, config
  mm = getMM(sharedMemSize)
  run = runserver(taskInSender, taskOutReceiver, noter, stopEvent, mm, isWindows)
  host = '127.0.0.1'
  port = config['port']
  if len(sys.argv) > 1:
    if '-g' in sys.argv:
      host = '0.0.0.0'
  else:
    from webbrowser import open as startBrowser
    from gevent import spawn_later
    spawn_later(1, startBrowser, 'http://127.0.0.1:{}'.format(port))
  run(host, port)