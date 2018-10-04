from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import scipy.misc as scipy
import os
import base64
from io import BytesIO
from runSR import dosr
from runDN import dodn
from video import SR_vid
import webbrowser
import threading
import os
import time
import cv2
app = Flask(__name__)

def index():
    return render_template('index.html')

index = app.route('/')(index)

def video():
    return render_template('video.html')

video = app.route('/video')(video)

def image_enhance():
    if request.method == 'POST':
        inputImg = request.files['file']
        scale = request.form['scale']
        mode = request.form['mode']
        denoise = request.form['denoise']
        cropsize = request.form['cropsize']
        if cropsize == '':
            cropsize = 256
        elif cropsize < '32':
            cropsize = 32
        image = scipy.misc.imread(inputImg)
        dnim = DN(image, cropsize, denoise)
        outputImg = SR(image, cropsize, scale, mode)
        outputImg = outputImg[..., ::-1]
        if not os.path.exists('download'):
            os.mkdir('download')
        cv2.imwrite('download/output.png', outputImg)
        return jsonify('Success', **None)

image_enhance = app.route('/image_enhance', [
    'GET',
    'POST'], **None)(image_enhance)

def SR(im, cropsize, scale, mode):
    if scale == '1':
        sim = im
    else:
        sim = dosr(im, int(scale), int(cropsize), mode)
    return sim


def DN(im, cropsize, denoise):
    if denoise == 'no':
        dim = im
    else:
        dim = dodn(im, int(cropsize), denoise)
    return dim


def downloader(filename):
    dirpath = os.path.join(app.root_path, 'download')
    return send_from_directory(dirpath, filename, True, **None)

downloader = app.route('/download/<path:filename>')(downloader)

def video_enhance():
    if request.method == 'POST':
        vidfile = request.files['file']
        scale = request.form['scale']
        mode = request.form['mode']
        denoise = request.form['denoise']
        cropsize = request.form['cropsize']
        if cropsize == '':
            cropsize = 256
        elif cropsize < '32':
            cropsize = 32
        vidname = vidfile.filename
        SR_vid(vidname, int(scale), int(cropsize), mode)
        return jsonify('Success', **None)

video_enhance = app.route('/video_enhance', [
    'GET',
    'POST'], **None)(video_enhance)

def runserver():
    app.debug = False
    app.run('127.0.0.1', 2333, **None)


def main_thread():
    thread1 = threading.Thread(start_server, **None)
    thread2 = threading.Thread(start_broswer, **None)
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

if __name__ == '__main__':
    main_thread()
