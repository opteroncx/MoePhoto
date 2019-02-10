import json
from io import BytesIO
from traceback import print_exc
from progress import setCallback, initialETA

def context(): pass
context.root = None
context.getFile = lambda size: BytesIO(context.sharedView[:size])

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
  }
  res.update(kwargs)
  if hasattr(node, 'name'):
    res['stage'] = node.name
    res['stageProgress'] = node.gone
    res['stageTotal'] = node.total
  context.notifier.send(res)

def enhance(f):
  def g(*args, **kwargs):
    try:
      result = f(*args, **kwargs)
      code = 200
    except Exception as e:
      print('错误内容=='+str(e))
      print_exc()
      result = 'Fail'
      code = 400
    finally:
      clean()
    return (json.dumps({'result': result}, ensure_ascii=False), code)
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
  print(routes)
  while True:
    task = taskIn.recv()
    print(task)
    stopEvent.clear()
    result = routes[task['name']](*task['args'], **task['kwargs']) if type(task) == dict else routes[task[0]](*task[1:])
    taskOut.send(result)