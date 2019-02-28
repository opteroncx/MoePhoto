import json
from io import BytesIO
from traceback import format_exc
from progress import setCallback, initialETA
from defaultConfig import defaultConfig
from logger import initLogging

def context(): pass
context.root = None
context.getFile = lambda size: BytesIO(context.sharedView[:size])
log = initLogging(defaultConfig['logPath'][0]).getLogger('Moe')

def filterOpt(item):
  if type(item) == dict and 'opt' in item:
    res = item.copy()
    del res['opt']
    return res
  else:
    return item

def begin(root, nodes=[], setAllCallback=True):
  context.root = root
  root.nodes = []
  for n in nodes:
    root.append(n)
  if setAllCallback:
    if not setAllCallback < 0:
      setCallback(root, onProgress, True)
  else:
    root.setCallback(onProgress)
  initialETA(root)
  return root

def onProgress(node, kwargs={}):
  res = {
    'eta': context.root.eta,
    'gone': context.root.gone,
    'total': context.root.total
  } if context.root else {}
  res.update(kwargs)
  if hasattr(node, 'name'):
    res['stage'] = node.name
    res['stageProgress'] = node.gone
    res['stageTotal'] = node.total
  context.notifier.send(res)

def enhance(f):
  def g(*args, **kwargs):
    try:
      res = { 'result': f(*args, **kwargs) }
      code = 200
    except:
      log.exception([f.__name__] + [filterOpt(arg) for arg in args])
      res = {
        'result': 'Fail',
        'exception': format_exc()
      }
      code = 400
    finally:
      clean()
    onProgress(context.root, res)
    return (json.dumps(res, ensure_ascii=False), code)
  return g

def worker(main, taskIn, taskOut, notifier, stopEvent):
  global clean, routes
  mm, routes = main()
  mm.seek(0)
  context.sharedView = memoryview(mm)
  context.shared = mm
  import imageProcess
  clean = imageProcess.clean
  context.notifier = notifier
  context.stopFlag = stopEvent
  while True:
    task = taskIn.recv()
    stopEvent.clear()
    result = routes[task[0]](*task[1:])
    taskOut.send(result)