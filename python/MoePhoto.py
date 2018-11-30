from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import webbrowser
import threading
import time
import codecs
import re
import psutil
import traceback
import readgpu
import imageProcess
from video import SR_vid, batchSR

app = Flask(__name__, root_path='.')
host = '127.0.0.1'
port = 2333

def system():
  mem = psutil.virtual_memory()
  mem_total = int(mem.total/1024**2)
  mem_free = int(mem.free/1024**2)
  cpu_count_phy = psutil.cpu_count(logical=False)
  cpu_count_log = psutil.cpu_count(logical=True)
  try:
    gname = readgpu.getName()[0].strip('\n')
    gram = readgpu.getGPU()
    ginfo = [gname,gram]
  except:
    gerror = '没有检测到NVIDIA的显卡，系统将采用CPU模式'
    ginfo = [gerror,gerror]
    print(gerror)
  return mem_total, mem_free, cpu_count_log, cpu_count_phy, ginfo

ndoc = '<a href="download/{image}" class="w3effct-agile"><img src="download/{image}"'+\
  ' alt="" class="img-responsive" title="Solar Panels Image" />'+\
  '<div class="agile-figcap"><h4>相册</h4><p>图片{image}</p></div></a>'

def gallery():
  items = ()
  try:
    items = os.listdir('download')
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

def enhance(f, file, *args):
  try:
    result = f(file, *args)
  except Exception as msg:
    print('错误内容=='+str(msg))
    traceback.print_exc()
    result = 'Fail'
  finally:
    imageProcess.clean()
  return jsonify(result=result)

def genNameByTime():
  itime = int(time.time())
  return 'download/output_{}.png'.format(itime)

@app.route('/image_enhance', methods=['POST'])
def image_enhance():
  scale = request.form['scale']
  mode = request.form['mode']
  denoise = request.form['denoise']
  dnseq = request.form['dn_seq']
  inputImg = request.files['file']
  process = imageProcess.genProcess(int(scale), mode, denoise, dnseq, 'file')
  return enhance(process, inputImg, genNameByTime())

@app.route("/download/<path:filename>")
def downloader(filename):
  dirpath = os.path.join(app.root_path, 'download')
  return send_from_directory(dirpath, filename, as_attachment=True)

@app.route('/video_enhance', methods=[
  'POST'])
def video_enhance():
  scale = request.form['scale']
  mode = request.form['mode']
  denoise = request.form['denoise']
  dnseq = request.form['dn_seq']
  codec = request.form['codec']
  vidfile = request.files['file']
  upload = 'upload'
  if not os.path.exists(upload):
    os.mkdir(upload)
  path ='{}/{}'.format(upload, vidfile.filename)
  vidfile.save(path)
  return enhance(SR_vid, path, int(scale), mode, denoise, dnseq, codec)

@app.route('/batch_enhance', methods=['POST'])
def batch_enhance():
  scale = request.form['scale']
  mode = request.form['mode']
  denoise = request.form['denoise']
  dnseq = request.form['dn_seq']
  fileList = request.files.getlist('filein')
  output_path = 'batch_SR/{}/'.format(int(time.time()))
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return enhance(batchSR, fileList, output_path, int(scale), mode, denoise, dnseq)

@app.route('/ednoise_enhance', methods=['POST'])
def ednoise_enhance():
  inputImg = request.files['file']
  denoise = request.form['denoise']
  process = imageProcess.genProcess(dnmodel=denoise, source='file')
  return enhance(process, inputImg, genNameByTime())

@app.route('/image_dehaze', methods=['POST'])
def image_dehaze():
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
  ('/system', 'system.html', None, system, ['mem_total', 'mem_free', 'cpu_count_log', 'cpu_count_phy', 'ginfo']),
  ('/gallery', 'gallery.html', None, gallery, ['var'])
]

app.route('/', endpoint='index')(lambda:render_template("index.html"))

def genPageFunction(item, header, footer):
  def page():
    res = item[3]()
    kargs = {}
    for k, v in zip(item[4], res):
      kargs[k] = v
    return render_template(item[1], header=header, footer=footer, **kargs)
  if item[3]:
    return page
  else:
    return lambda:render_template(item[1], header=header, footer=footer)

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
  app.run(host, port)

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
  if (len(sys.argv) > 1) and ('-n' in sys.argv):
    runserver()
  else:
    main_thread()
