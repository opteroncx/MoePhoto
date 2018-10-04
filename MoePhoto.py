from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import scipy.misc
import os
import base64
from io import BytesIO
from runSR import dosr
from runDN import dodn
from video import SR_vid
from dehaze import Dehaze
import webbrowser
import threading
import os
import time
import jinja2
import cv2
import codecs
import re
import psutil
import readgpu

app = Flask(__name__)

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


@app.route('/image_enhance', methods=['GET', 'POST'])
def image_enhance():
    if request.method == "POST":
        inputImg = request.files['file']
        scale = request.form['scale']
        mode = request.form['mode']
        denoise = request.form['denoise']
        dnseq = request.form['dn_seq']

        image = scipy.misc.imread(inputImg)
        if dnseq == 'before':
            dnim = DN(image, denoise)
            outputImg = SR(image, scale, mode)
        elif dnseq == 'after':
            sim = SR(image, scale, mode)
            outputImg = DN(sim, denoise)
        outputImg = outputImg[..., ::-1]
        if not os.path.exists('download'):
            os.mkdir('download')
        itime = str(int(time.time()))
        iname = 'download/output_'+itime+'.png'
        siname = '/'+iname+'?'
        cv2.imwrite(iname, outputImg)       
        return jsonify(result=siname)


def SR(im, scale, mode):
    if scale == "1":
        sim = im
    else:
        sim = dosr(im, int(scale), mode)
    return sim


def DN(im, denoise):
    if denoise == "no":
        dim = im
    else:
        dim = dodn(im, denoise)
    return dim


@app.route("/download/<path:filename>")
def downloader(filename):
    dirpath = os.path.join(app.root_path, 'download')
    return send_from_directory(dirpath, filename, as_attachment=True)


@app.route('/video_enhance', methods=['GET', 'POST'])
def video_enhance():
    if request.method == "POST":
        vidfile = request.files['file']
        scale = request.form['scale']
        mode = request.form['mode']
        denoise = request.form['denoise']
        dnseq = request.form['dn_seq']
        upload = 'upload'
        if not os.path.exists(upload):
            os.mkdir(upload)
        vidname = vidfile.filename
        ext = vidname.rsplit('.', 1)[1]
        vidfile.save(upload+'/video.'+ext)
        SR_vid(upload+'/video.'+ext, int(scale), mode, denoise, dnseq)
        return jsonify(result="Success")


@app.route('/batch_enhance', methods=['GET', 'POST'])
def batch_enhance():
    if request.method == "POST":
        scale = request.form['scale']
        mode = request.form['mode']
        denoise = request.form['denoise']
        dnseq = request.form['dn_seq']
        output_path = 'batch_SR/'+str(int(time.time()))+'/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        count = 1
        for inputImg in request.files.getlist('filein'):
            print(inputImg)
            image = scipy.misc.imread(inputImg)
            if dnseq == 'before':
                dnim = DN(image, denoise)
                outputImg = SR(image, scale, mode)
            elif dnseq == 'after':
                sim = SR(image, scale, mode)
                outputImg = DN(sim, denoise)
            outputImg = outputImg[..., ::-1]
            cv2.imwrite(output_path+str(count)+'.png', outputImg)
            count += 1
        return jsonify(result="Success")

@app.route('/ednoise_enhance', methods=['GET', 'POST'])
def ednoise_enhance():
    if request.method == "POST":
        inputImg = request.files['file']
        denoise = request.form['denoise']
        image = scipy.misc.imread(inputImg)
        outputImg = DN(image, denoise)
        outputImg = outputImg[..., ::-1]
        if not os.path.exists('download'):
            os.mkdir('download')
        itime = str(int(time.time()))
        iname = 'download/output_'+itime+'.png'
        siname = '/'+iname+'?'
        cv2.imwrite(iname, outputImg)       
        return jsonify(result=siname)

@app.route('/image_dehaze', methods=['GET', 'POST'])
def image_dehaze():
    if request.method == "POST":
        inputImg = request.files['file']
        image = scipy.misc.imread(inputImg)
        outputImg = Dehaze(image)
        outputImg = outputImg[..., ::-1]
        if not os.path.exists('download'):
            os.mkdir('download')
        itime = str(int(time.time()))
        iname = 'download/output_'+itime+'.png'
        siname = '/'+iname+'?'
        cv2.imwrite(iname, outputImg)       
        return jsonify(result=siname)

def runserver():
    app.debug = False
    app.run(host="127.0.0.1", port=2333)

def main_thread():
    thread1 = threading.Thread(target=start_server,)
    thread2 = threading.Thread(target=start_broswer,)
    thread1.setDaemon(True)
    thread1.start()

    time.sleep(2)
    thread2.start()

    thread1.join()
    thread2.join()


def start_server():
    runserver()


def start_broswer():
    url = 'http://127.0.0.1:2333'
    webbrowser.open(url)


if __name__ == "__main__":
    main_thread()
    # runserver()

