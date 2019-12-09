import json
from io import BytesIO
from traceback import format_exc
from gevent import idle
from progress import setCallback, initialETA, saveOps, loadOps
from config import config
from logger import initLogging

def context(): pass
context.root = None
context.getFile = lambda size: BytesIO(context.sharedView[:size])
log = initLogging(config.logPath).getLogger('Moe') # pylint: disable=E1101
opsPath = config.opsPath # pylint: disable=E1101
getInfo = lambda f, args: [f.__name__] + [filterOpt(arg) for arg in args]

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
  saveOps(opsPath)
  if hasattr(node, 'name'):
    res['stage'] = node.name
    res['stageProgress'] = node.gone
    res['stageTotal'] = node.total
  context.notifier.send(res)

def enhance(f, verbose=True):
  def g(*args, **kwargs):
    try:
      res = { 'result': f(*args, **kwargs) }
      code = 200
      saveOps(opsPath, True)
      if verbose:
        log.info(getInfo(f, args))
    except:
      info = getInfo(f, args)
      log.exception(info)
      res = {
        'result': 'Fail',
        'call': info,
        'exception': format_exc()
      }
      code = 400
      context.notifier.send(res)
    finally:
      clean()
    return res, code
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
  loadOps(opsPath)
  while True:
    idle()
    task = taskIn.recv()
    stopEvent.clear()
    result = routes[task[0]](*task[1:])
    taskOut.send(result)