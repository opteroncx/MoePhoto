import os
import time
import json
import codecs
import re
import psutil
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response, Response, send_file
from gevent import pywsgi, idle, spawn
from defaultConfig import defaultConfig

staticMaxAge = 86400
app = Flask(__name__, root_path='.')
app.config['SERVER_NAME'] = '.'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = staticMaxAge
startupTime = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
def current():pass
current.session = None
current.path = None
current.eta = 0
current.fileSize = 0
E403 = ('Not authorized.', 403)
E404 = ('Not Found', 404)
OK = ('', 200)
busy = lambda: (jsonify(result='Busy', eta=current.eta), 503)
cwd = os.getcwd()
outDir = defaultConfig['outDir'][0]
uploadDir = defaultConfig['uploadDir'][0]
logPath = os.path.abspath(defaultConfig['logPath'][0])
downDir = os.path.join(app.root_path, outDir)
if not os.path.exists(outDir):
  os.mkdir(outDir)
with open('static/manifest.json') as manifest:
  assetMapping = json.load(manifest)
vendorsJs = assetMapping['vendors.js']
commonJs = assetMapping['common.js'] if 'common.js' in assetMapping else None

def acquireSession(request):
  if current.session:
    return busy()
  while noter.poll():
    noter.recv()
  current.session = request.values['session']
  current.path = request.path
  current.eta = 10
  return False if current.session else E403

def controlPoint(path, fMatch, fUnmatch, resNoCurrent, check=lambda *_: True):
  def f():
    try:
      session = request.values['session']
    except:
      return E403
    if not session:
      return E403
    if current.session:
      return spawn(fMatch).get() if current.session == session and check(request) else fUnmatch()
    else:
      return resNoCurrent
  app.route(path, methods=['GET', 'POST'], endpoint=path)(f)

def stopCurrent():
  if current.session:
    current.stopFlag.set()
  return OK

def checkMsgMatch(request):
  if not 'path' in request.values:
    return True
  path = request.values['path']
  return path == current.path

def onConnect():
  while current.session and not noter.poll():
    idle()
  if current.session and noter.poll():
    while noter.poll():
      res = noter.recv()
    current.eta = res['eta']
    if 'fileSize' in res:
      current.fileSize = res['fileSize']
      del res['fileSize']
    return (json.dumps(res, ensure_ascii=False), 200)
  else:
    return OK

def makeHandler(name, prepare, final, methods=['POST']):
  def f():
    c = acquireSession(request)
    if c:
      return c
    sender.send((name, *prepare(request)))
    while not receiver.poll():
      idle()
    result = receiver.recv()
    current.session = None
    return final(result)
  app.route('/' + name, methods=methods, endpoint=name)(f)

def renderPage(item, header=None, footer=None):
  other = item[5] if len(item) > 5 else {}
  other['vendorsJs'] = vendorsJs
  if commonJs:
    other['commonJs'] = commonJs
  template = item[1]
  func = item[3]
  if func:
    g = lambda req: render_template(
      template, header=header, footer=footer, **other, **dict(zip(item[4], func(req))))
  else:
    with app.app_context():
      cache = render_template(template, header=header, footer=footer, **other)
    g = lambda _: cache
  def f():
    session = request.cookies.get('session')
    resp = make_response(g(request))
    t = time.time()
    if (not session) or (float(session) > t):
      resp.set_cookie('session', bytes(str(t), encoding='ascii'))
    if func:
      resp.cache_control.private = True
    else:
      resp.headers['Last-Modified'] = startupTime
      resp.cache_control.max_age = staticMaxAge
    return resp
  return f

ndoc = '<a href="{dirName}/{image}" class="w3effct-agile"><img src="{dirName}/{image}"'+\
  ' alt="" class="img-responsive" title="Solar Panels Image" />'+\
  '<div class="agile-figcap"><h4>相册</h4><p>图片{image}</p></div></a>'

def gallery(req):
  items = ()
  dirName = req.values['dir'] if 'dir' in req.values else outDir
  try:
    items = os.listdir(dirName)
  except:pass
  images = filter((lambda item:item.endswith('.png') or item.endswith('.jpg')), items)
  doc = []
  images = [*map(lambda image:ndoc.format(image=image, dirName=dirName), images)]
  for i in range((len(images) - 1) // 3 + 1):
    doc.append('<div class="col-sm-4 col-xs-4 w3gallery-grids">')
    doc.extend(images[i * 3:(i + 1) * 3])
    doc.append('</div>')
  return (''.join(doc),) if len(doc) else ('暂时没有图片，快去尝试放大吧',)

def getSystemInfo():
  import readgpu
  cuda, cudnn = readgpu.getCudaVersion()
  info = {
    'cpu_count_phy': psutil.cpu_count(logical=False),
    'cpu_count_log': psutil.cpu_count(logical=True),
    'cpu_freq': psutil.cpu_freq().max,
    'disk_total': psutil.disk_usage(cwd).total // 2**20,
    'mem_total': psutil.virtual_memory().total // 2**20,
    'python': readgpu.getPythonVersion(),
    'torch': readgpu.getTorchVersion(),
    'cuda': cuda,
    'cudnn': cudnn,
    'gpus': readgpu.getGPUProperties()
  }
  readgpu.uninstall()
  del readgpu
  return info

def getDynamicInfo(_):
  disk_free = psutil.disk_usage(cwd).free // 2**20
  mem_free = psutil.virtual_memory().free // 2**20
  return disk_free, mem_free, current.session, current.path

about_updater = lambda *_: [codecs.open('./update_log.txt', encoding='utf-8').read()]

header = codecs.open('./templates/1-header.html','r','utf-8').read()
footer = codecs.open('./templates/1-footer.html','r','utf-8').read()
routes = [
  #(query path, template file, active page name, request handler, request result names, dict of static variables)
  ('/', 'index.html', '主页', None),
  ('/video', 'video.html', 'AI视频', None),
  ('/batch', 'batch.html', '批量放大', None),
  ('/ednoise', 'ednoise.html', '风格化', None),
  ('/dehaze', 'dehaze.html', '去雾', None),
  ('/document', 'document.html', None, None),
  ('/about', 'about.html', None, about_updater, ['log']),
  ('/system', 'system.html', None, getDynamicInfo, ['disk_free', 'mem_free', 'session', 'path'], getSystemInfo()),
  ('/gallery', 'gallery.html', None, gallery, ['var'])
]

for item in routes:
  if item[2]:
    pattern = '>' + item[2]
    new = 'class=\"active\"' + pattern
    h = re.sub(pattern,new,header)
  else:
    h = header
  app.route(item[0], endpoint=item[0])(renderPage(item, h, footer))

identity = lambda x: x
readOpt = lambda req: json.loads(req.values['steps'])
controlPoint('/stop', stopCurrent, lambda: E403, E404)
controlPoint('/msg', onConnect, busy, OK, checkMsgMatch)
app.route('/log', endpoint='log')(lambda: send_file(logPath, add_etags=False))
app.route('/favicon.ico', endpoint='favicon')(lambda: send_from_directory(app.root_path, 'logo3.ico'))
app.route("/{}/.preview.png".format(outDir), endpoint='preview')(lambda: Response(current.getPreview(), mimetype='image/png'))
sendFromDownDir = lambda filename: send_from_directory(downDir, filename, as_attachment=True)
app.route("/{}/<path:filename>".format(outDir), endpoint='download')(sendFromDownDir)
lockFinal = lambda result: (jsonify(result='Interrupted', remain=result), 499) if result > 0 else OK
makeHandler('lock', (lambda req: [int(req.values['duration'])]), lockFinal, ['GET', 'POST'])
makeHandler('systemInfo', (lambda _: []), identity, ['GET', 'POST'])
imageEnhancePrep = lambda req: (current.writeFile(req.files['file']), *readOpt(req))
makeHandler('image_enhance', imageEnhancePrep, identity)
makeHandler('ednoise_enhance', imageEnhancePrep, identity)
makeHandler('image_dehaze', lambda req: (current.writeFile(req.files['file']), {'op': 'dehaze'}), identity)

def videoEnhancePrep(req):
  vidfile = req.files['file']
  if not os.path.exists(uploadDir):
    os.mkdir(uploadDir)
  path ='{}/{}'.format(uploadDir, vidfile.filename)
  vidfile.save(path)
  return (path, *readOpt(req))
makeHandler('video_enhance', videoEnhancePrep, identity)

@app.route('/batch_enhance', methods=['POST'])
def batchEnhance():
  c = acquireSession(request)
  if c:
    return c
  count = 0
  fail = 0
  result = 'Success'
  fileList = request.files.getlist('file')
  output_path = '{}/{}/'.format(outDir, int(time.time()))
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  opt = readOpt(request)
  total = len(fileList)
  print('batch total: {}'.format(total))
  current.notifier.send({
    'eta': total,
    'gone': 0,
    'total': total
  })
  opt[-1]['trace'] = False
  for image in fileList:
    if current.stopFlag.is_set():
      result = 'Interrupted'
      break
    name = output_path + os.path.basename(image.filename)
    start = time.time()
    opt[-1]['file'] = name
    sender.send(('image_enhance', current.writeFile(image), *opt))
    while not receiver.poll():
      idle()
    if receiver.poll():
      output = receiver.recv()
      count += 1
      note = {
        'eta': (total - count) * (time.time() - start),
        'gone': count,
        'total': total
      }
      if output[1] == 200:
        note['preview'] = name
      else:
        fail += 1
      current.notifier.send(note)
  current.session = None
  return json.dumps({'result': (result, count, fail, output_path)}, ensure_ascii=False)

def runserver(taskInSender, taskOutReceiver, noteReceiver, stopEvent, notifier, mm):
  global sender, receiver, noter
  sender = taskInSender
  receiver = taskOutReceiver
  noter = noteReceiver
  current.notifier = notifier
  current.stopFlag = stopEvent
  def preview():
    mm.seek(0)
    buffer = mm.read(current.fileSize)
    return buffer
  current.getPreview = preview
  def writeFile(file):
    mm.seek(0)
    return file._file.readinto(mm)
  current.writeFile = writeFile
  def f(host, port):
    app.debug = False
    app.config['SERVER_NAME'] = None
    server = pywsgi.WSGIServer((host, port), app)
    print('Current working directory: {}'.format(cwd))
    print('Server starts to listen on http://{}:{}/, press Ctrl+C to exit.'.format(host, port))
    server.serve_forever()
  return f