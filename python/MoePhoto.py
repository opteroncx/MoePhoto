import sys
import multiprocessing as mp
from mmap import mmap
from defaultConfig import defaultConfig
sharedMemSize = defaultConfig['sharedMemSize'][0]
isWindows = sys.platform[:3] == 'win'
mm = mmap(-1, sharedMemSize, tagname='SharedMemory') if isWindows else mmap(-1, sharedMemSize)

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
    trace = outputOpt['trace'] if 'trace' in outputOpt else True
    process, nodes = genProcess(stepFile + list(args))
    return begin(imNode, nodes, trace).bindFunc(process)(size, name=name)

  return mm, {
    'lockInterface': lock,
    'image_enhance': enhance(imageEnhance, verbose=False),
    'batch': enhance(imageEnhance, verbose=False),
    'video_enhance': enhance(SR_vid),
    'systemInfo': enhance(config.system)
  }

if __name__ == '__main__':
  mp.freeze_support()
  from worker import worker
  taskInReceiver, taskInSender = mp.Pipe(False)
  taskOutReceiver, taskOutSender = mp.Pipe(False)
  noter, notifier = mp.Pipe(False)
  stopEvent = mp.Event()
  mp.Process(target=worker, args=(main, taskInReceiver, taskOutSender, notifier, stopEvent), daemon=True).start()
  from server import runserver, config
  run = runserver(taskInSender, taskOutReceiver, noter, stopEvent, mm)
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