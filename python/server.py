import os
import time
import json
import codecs
import re
import psutil
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response, Response
from gevent import pywsgi, idle, spawn
from defaultConfig import defaultConfig

staticMaxAge = 86400
app = Flask(__name__, root_path='.')
app.config['SERVER_NAME'] = '.'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = staticMaxAge
startupTime = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
def current():pass
current.session = None
current.eta = 0
current.fileSize = 0
E403 = ('Not authorized.', 403)
E404 = ('Not Found', 404)
OK = ('', 200)
busy = lambda: (jsonify(result='Busy', eta=current.eta), 503)
cwd = os.getcwd()
outDir = defaultConfig['outDir'][0]
uploadDir = defaultConfig['uploadDir'][0]
downDir = os.path.join(app.root_path, outDir)
if not os.path.exists(outDir):
  os.mkdir(outDir)

def acquireSession(request):
  if current.session:
    return busy()
  current.session = request.values['session']
  current.eta = 10
  return False if current.session else E403

def controlPoint(path, fMatch, fUnmatch, resNoCurrent):
  def f():
    try:
      session = request.values['session']
    except:
      return E403
    if not session:
      return E403
    if current.session:
      return spawn(fMatch).get() if current.session == session else fUnmatch()
    else:
      return resNoCurrent
  app.route(path, methods=['GET', 'POST'], endpoint=path)(f)

def stopCurrent():
  if current.session:
    current.stopFlag.set()
  return OK

def onConnect():
  while current.session and not noter.poll():
    idle()
  if current.session and noter.poll():
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

def renderPage(template, header=None, footer=None, dynamic=False, other={}):
  if dynamic:
    g = lambda ks: render_template(template, header=header, footer=footer, **other, **ks)
  else:
    with app.app_context():
      cache = render_template(template, header=header, footer=footer, **other)
    g = lambda _: cache
  def f(ks={}):
    session = request.cookies.get('session')
    resp = make_response(g(ks))
    t = time.time()
    if (not session) or (float(session) > t):
      resp.set_cookie('session', bytes(str(t), encoding='ascii'))
    if dynamic:
      resp.cache_control.private = True
    else:
      resp.headers['Last-Modified'] = startupTime
      resp.cache_control.max_age = staticMaxAge
    return resp
  return f

def genPageFunction(item, header, footer):
  r = renderPage(item[1], header, footer, item[3], item[5] if len(item) > 5 else {})
  page = lambda: r(dict(zip(item[4], item[3]())))
  return page if item[3] else r

ndoc = '<a href="download/{image}" class="w3effct-agile"><img src="download/{image}"'+\
  ' alt="" class="img-responsive" title="Solar Panels Image" />'+\
  '<div class="agile-figcap"><h4>相册</h4><p>图片{image}</p></div></a>'

def gallery():
  items = ()
  try:
    items = os.listdir(outDir)
  except:pass
  images = filter((lambda item:item.endswith('.png')), items)
  doc = []
  images = [*map(lambda image:ndoc.format(image=image), images)]
  for i in range((len(images) - 1) // 3 + 1):
    doc.append('<div class="col-sm-4 col-xs-4 w3gallery-grids">')
    doc.extend(images[i * 3:(i + 1) * 3])
    doc.append('</div>')
  return (''.join(doc),) if len(doc) else ('暂时没有图片，快去尝试放大吧',)

def getSystemInfo():
  import readgpu
  info = {
    'cpu_count_phy': psutil.cpu_count(logical=False),
    'cpu_count_log': psutil.cpu_count(logical=True),
    'cpu_freq': psutil.cpu_freq().max,
    'disk_total': psutil.disk_usage(cwd).total // 2**20,
    'mem_total': psutil.virtual_memory().total // 2**20,
    'python': readgpu.getPythonVersion(),
    'torch': readgpu.getTorchVersion(),
    'gpus': readgpu.getGPUProperties()
  }
  readgpu.uninstall()
  del readgpu
  return info

def getDynamicInfo():
  disk_free = psutil.disk_usage(cwd).free // 2**20
  mem_free = psutil.virtual_memory().free // 2**20
  return disk_free, mem_free

header = codecs.open('./templates/1-header.html','r','utf-8').read()
footer = codecs.open('./templates/1-footer.html','r','utf-8').read()
routes = [
  ('/video', 'video.html', '视频放大', None),
  ('/batch', 'batch.html', '批量放大', None),
  ('/ednoise', 'ednoise.html', '风格化', None),
  ('/deblur', 'deblur.html', '去模糊', None),
  ('/dehaze', 'dehaze.html', '去雾', None),
  ('/document', 'document.html', None, None),
  ('/about', 'about.html', None, None),
  ('/system', 'system.html', None, getDynamicInfo, ['disk_free', 'mem_free'], getSystemInfo()),
  ('/gallery', 'gallery.html', None, gallery, ['var'])
]

for item in routes:
  if item[2]:
    pattern = '>' + item[2]
    new = 'class=\"active\"' + pattern
    h = re.sub(pattern,new,header)
  else:
    h = header
  app.route(item[0], endpoint=item[0])(genPageFunction(item, h, footer))

optKeys = ('scale', 'mode', 'denoise', 'dn_seq')
def readOpt(req):
  t = [req.values[key] if key in req.values else 0 for key in optKeys]
  t[0] = int(t[0])
  return t

identity = lambda x: x
controlPoint('/stop', stopCurrent, lambda: E403, E404)
controlPoint('/msg', onConnect, busy, OK)
app.route('/', endpoint='index')(renderPage("index.html"))
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
makeHandler('image_dehaze', imageEnhancePrep, identity)

def videoEnhancePrep(req):
  opt = req.form.copy()
  vidfile = req.files['file']
  if not os.path.exists(uploadDir):
    os.mkdir(uploadDir)
  path ='{}/{}'.format(uploadDir, vidfile.filename)
  vidfile.save(path)
  return (path, opt)
makeHandler('video_enhance', videoEnhancePrep, identity)

def batchEnhancePrep(req):
  fileList = req.files.getlist('filein')
  output_path = '{}/{}/'.format(outDir, int(time.time()))
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return (fileList, output_path, *readOpt(req))
makeHandler('batch_enhance', batchEnhancePrep, identity)

def runserver(taskInSender, taskOutReceiver, noteReceiver, stopEvent, mm):
  global sender, receiver, noter
  sender = taskInSender
  receiver = taskOutReceiver
  noter = noteReceiver
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
    app.config['SERVER_NAME'] = '{}:{}'.format(host, port)
    server = pywsgi.WSGIServer((host, port), app)
    print('Current working directory: {}'.format(cwd))
    print('Server starts to listen on http://{}:{}/, press Ctrl+C to exit.'.format(host, port))
    server.serve_forever()
  return f