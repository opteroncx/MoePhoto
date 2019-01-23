from gevent import pywsgi, monkey, idle, spawn
monkey.patch_select()
monkey.patch_socket()
from gevent.event import Event
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response, Response
import json
import os
import sys
import webbrowser
import threading
import time
import codecs
import re
import traceback
import imageProcess
from video import SR_vid, batchSR
from config import config
from progress import initialETA, Node, setCallback

app = Flask(__name__, root_path='.')
host = '127.0.0.1'
port = 2333
def current():pass
current.session = None
current.response = None
current.root = None
current.previewIm = None
E403 = ('Not authorized.', 403)
E404 = ('Not Found', 404)
OK = ('', 200)
pending = Event()
sent = Event()
sent.set()
outDir = 'download'

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
  if len(doc):
    return (''.join(doc),)
  else:
    return ('暂时没有图片，快去尝试放大吧',)

busy = lambda: (jsonify(result='Busy', eta=current.root.eta), 503)
genNameByTime = lambda: 'download/output_{}.png'.format(int(time.time()))

def acquireSession(request):
  if current.session:
    return busy()
  current.session = request.values['session']
  return False if current.session else E403

def notify(msg):
  if current.session:
    current.response = msg
    if not pending.isSet():
      sent.clear()
      pending.set()
      sent.wait()
    idle()

def onProgress(node, kwargs={}):
  res = {
    'eta': current.root.eta,
    'gone': current.root.gone,
    'total': current.root.total
  }
  res.update(kwargs)
  if hasattr(node, 'name'):
    res['stage'] = node.name
    res['stageProgress'] = node.gone
    res['stageTotal'] = node.total
  notify(json.dumps(res, ensure_ascii=False))

def begin(root, nodes, setAllCallback=True):
  current.root = root
  for n in nodes:
    root.append(n)
  if setAllCallback:
    setCallback(root, onProgress)
  else:
    root.setCallback(onProgress)
  initialETA(root)
  root.running = True
  return root, current

def controlPoint(path, fMatch, fUnmatch, resNoCurrent):
  def f():
    try:
      session = request.values['session']
    except:
      return E403
    if not session:
      return E403
    if current.session:
      if current.session == session:
        return fMatch()
      else:
        return fUnmatch()
    else:
      return resNoCurrent
  app.route(path, methods=['GET', 'POST'], endpoint=path)(f)

def stopCurrent():
  if current.root:
    current.root.running = False
    print('STOP on {} of {}'.format(current.root.gone, current.root.total)) #debug
  return OK

def onConnect():
  pending.wait()
  pending.clear()
  sent.set()
  return (current.response, 200)

controlPoint('/stop', stopCurrent, lambda: E403, E404)
controlPoint('/msg', onConnect, busy, OK)

@app.route('/lock', methods=['GET', 'POST'])
def testFunc():
  c = acquireSession(request)
  if c:
    return c
  duration = int(request.values['duration'])
  node = begin(Node({}, 1, duration, 0), [])[0].reset()
  flag = Event()
  while duration > 0 and node.running:
    duration -= 1
    flag.wait(1)
    flag.clear()
    node.trace()
  current.session = None
  return (jsonify(result='Interrupted'), 499) if duration > 0 else OK

def enhance(f, file, *args):
  try:
    g = spawn(f, file, *args)
    result = g.get()
    code = 200
  except Exception as msg:
    print('错误内容=='+str(msg))
    traceback.print_exc()
    result = 'Fail'
    code = 400
  finally:
    imageProcess.clean()
    current.session = None
  return (jsonify(result=result), code)

@app.route('/image_enhance', methods=['POST'])
def image_enhance():
  c = acquireSession(request)
  if c:
    return c
  scale = request.form['scale']
  mode = request.form['mode']
  denoise = request.form['denoise']
  dnseq = request.form['dn_seq']
  inputImg = request.files['file']
  process, nodes = imageProcess.genProcess(int(scale), mode, denoise, dnseq, 'file')
  return enhance(begin(Node({'op': 'image'}, learn=0), nodes)[0].bindFunc(process), inputImg, genNameByTime())

@app.route("/{}/<path:filename>".format(outDir))
def downloader(filename):
  dirpath = os.path.join(app.root_path, outDir)
  return send_from_directory(dirpath, filename, as_attachment=True)

app.route("/{}/.preview.png".format(outDir))(lambda: Response(current.previewIm, mimetype='image/png'))

@app.route('/video_enhance', methods=[
  'POST'])
def video_enhance():
  c = acquireSession(request)
  if c:
    return c
  opt = request.form.copy()
  opt['outDir'] = outDir
  vidfile = request.files['file']
  upload = 'upload'
  if not os.path.exists(upload):
    os.mkdir(upload)
  path ='{}/{}'.format(upload, vidfile.filename)
  vidfile.save(path)
  return enhance(SR_vid, path, begin, opt)

@app.route('/batch_enhance', methods=['POST'])
def batch_enhance():
  c = acquireSession(request)
  if c:
    return c
  scale = request.form['scale']
  mode = request.form['mode']
  denoise = request.form['denoise']
  dnseq = request.form['dn_seq']
  fileList = request.files.getlist('filein')
  output_path = 'batch_SR/{}/'.format(int(time.time()))
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return enhance(batchSR, fileList, output_path, int(scale), mode, denoise, dnseq, begin)

@app.route('/ednoise_enhance', methods=['POST'])
def ednoise_enhance():
  c = acquireSession(request)
  if c:
    return c
  inputImg = request.files['file']
  denoise = request.form['denoise']
  process, _ = imageProcess.genProcess(dnmodel=denoise, source='file')
  return enhance(process, inputImg, genNameByTime())

@app.route('/image_dehaze', methods=['POST'])
def image_dehaze():
  c = acquireSession(request)
  if c:
    return c
  inputImg = request.files['file']
  return enhance(imageProcess.dehaze, inputImg, genNameByTime())
routes = [
  ('/video', 'video.html', '视频放大', None),
  ('/batch', 'batch.html', '批量放大', None),
  ('/ednoise', 'ednoise.html', '风格化', None),
  ('/deblur', 'deblur.html', '去模糊', None),
  ('/dehaze', 'dehaze.html', '去雾', None),
  ('/document', 'document.html', None, None),
  ('/about', 'about.html', None, None),
  ('/system', 'system.html', None, config.system, ['mem_total', 'mem_free', 'cpu_count_log', 'cpu_count_phy', 'ginfo']),
  ('/gallery', 'gallery.html', None, gallery, ['var'])
]

def renderPage(template, header=None, footer=None):
  def f(ks={}):
    session = request.cookies.get('session')
    resp = make_response(render_template(template, header=header, footer=footer, **ks))
    t = time.time()
    if (not session) or (float(session) > t):
      resp.set_cookie('session', bytes(str(t), encoding='ascii'))
    return resp
  return f

app.route('/', endpoint='index')(renderPage("index.html"))
app.route('/favicon.ico', endpoint='favicon')(lambda:send_from_directory(app.root_path, 'logo3.ico'))

def genPageFunction(item, header, footer):
  r = renderPage(item[1], header, footer)
  def page():
    res = item[3]()
    kargs = {}
    for k, v in zip(item[4], res):
      kargs[k] = v
    return r(kargs)
  return page if item[3] else r

header_html = codecs.open('./templates/1-header.html','r','utf-8')
footer_html = codecs.open('./templates/1-footer.html','r','utf-8')
header = header_html.read()
footer = footer_html.read()

for item in routes:
  if item[2]:
    pattern = '>' + item[2]
    new = 'class=\"active\"' + pattern
    h = re.sub(pattern,new,header)
  else:
    h = header
  app.route(item[0], endpoint=item[0])(genPageFunction(item, h, footer))

def runserver():
  app.debug = False
  server = pywsgi.WSGIServer((host, port), app)
  print('Server starts to listen on http://{}:{}/, press Ctrl+C to exit.'.format(host, port))
  server.serve_forever()

def main_thread():
  thread1 = threading.Thread(target=runserver,)
  thread2 = threading.Thread(target=start_broswer,)
  thread1.setDaemon(True)
  thread1.start()

  threading.Event().wait(2)
  thread2.start()

  thread1.join()
  thread2.join()

def start_broswer():
  url = 'http://localhost:{}'.format(port)
  webbrowser.open(url)

if __name__ == "__main__":
  if not os.path.exists(outDir):
    os.mkdir(outDir)
  if len(sys.argv) > 1:
    if '-g' in sys.argv:
      host = ''
    runserver()
  else:
    main_thread()
