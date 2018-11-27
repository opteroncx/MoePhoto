from flask import Flask, render_template, request, jsonify, send_from_directory
import os
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

def get_head_foot(cls=None):
  header_html = codecs.open('./templates/1-header.html','r','utf-8')
  footer_html = codecs.open('./templates/1-footer.html','r','utf-8')
  header = header_html.read()
  footer = footer_html.read()
  if cls != None:
    pattern = '>'+cls
    new = 'class=\"active\"'+pattern
    header = re.sub(pattern,new,header)
  return header,footer

@app.route('/')
def index():
  return render_template("index.html")

@app.route('/video')
def video():
  header,footer = get_head_foot('视频放大')
  return render_template("video.html",header=header,footer=footer)

@app.route('/batch')
def batch():
  header,footer = get_head_foot('批量放大')
  return render_template("batch.html",header=header,footer=footer)

@app.route('/ednoise')
def ednoise():
  header,footer = get_head_foot('风格化')
  return render_template("ednoise.html",header=header,footer=footer)

@app.route('/deblur')
def deblur():
  header,footer = get_head_foot('去模糊')
  return render_template("deblur.html",header=header,footer=footer)

@app.route('/dehaze')
def dehaze():
  header,footer = get_head_foot('去雾')
  return render_template("dehaze.html",header=header,footer=footer)

@app.route('/document')
def document():
  header,footer = get_head_foot()
  return render_template("document.html",header=header,footer=footer)

@app.route('/about')
def about():
  header,footer = get_head_foot()
  return render_template("about.html",header=header,footer=footer)

@app.route('/system')
def system():
  header,footer = get_head_foot()
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
    print(ginfo)
  return render_template("system.html",header=header,footer=footer,\
      mem_total=mem_total,mem_free=mem_free,cpu_count_log=cpu_count_log,\
      cpu_count_phy=cpu_count_phy,ginfo=ginfo)

@app.route('/gallery')
def gallery():
  header,footer = get_head_foot()
  path = 'download/'
  doc = ''
  try:
    items = os.listdir(path)
    images = []
    for item in items:
      if item[-4:] == '.png':
        images.append(item)
    for i in range(len(images)):
      if i % 3 == 0:
        doc += '<div class="col-sm-4 col-xs-4 w3gallery-grids">'
      ndoc = '<a href=\"'+path+images[i]+'" class=\"w3effct-agile\">'+'<img src='+path+images[i]+\
        ' alt="" class=\"img-responsive\" title=\"Solar Panels Image\" />'+\
        '<div class=\"agile-figcap\"><h4>相册</h4><p>图片'+images[i]+'</p></div></a>'
      doc+=ndoc
      if i % 3 == 0:
        doc+='</div>'
  except:
    doc+='暂时没有图片，快去尝试放大吧'
  page = render_template("gallery.html", var=doc,header=header,footer=footer)
  return page

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

@app.route('/image_enhance', methods=[
  'POST'])
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
  #main_thread()
  runserver()
