import os
import subprocess as sp
import re
import sys
import threading
from queue import Queue, Empty
from config import Config
import imageProcess

ffmpegPath = os.path.realpath('ffmpeg/bin/ffmpeg') # require full path to spawn in shell
defaultCodec = 'libx264 -pix_fmt yuv420p'
qOut = Queue(64)

def getVideoInfo(videoPath):
  commandIn = [
    ffmpegPath,
    '-i', videoPath
  ]
  try:
    pipeIn = sp.Popen(commandIn, stderr=sp.PIPE, encoding='utf_8')

    for line in iter(pipeIn.stderr.readline, ''):
      line = line.lstrip()
      if re.match('Stream #.*: Video:', line):
        try:
          videoInfo = re.search(', ([\\d]+)x([\\d]+) *.+, ([.\\d]+) fps', line).groups()
          width = int(videoInfo[0])
          height = int(videoInfo[1])
          frameRate = float(videoInfo[2])
        except:
          print(line)
          raise RuntimeError('Video info not found')
        break

    pipeIn.stderr.flush()
    pipeIn.stderr.close()
  finally:
    pipeIn.terminate()
  return width, height, frameRate

def batchSR(images, srpath, scale, mode, dnmodel, dnseq):
  count = 0
  process = imageProcess.genProcess(scale, mode, dnmodel, dnseq, 'file')
  for image in images:
    count += 1
    print('processing image {}'.format(image.filename))
    fileName = '{}{}.png'.format(srpath, count)
    try:
      process(image, fileName)
    except Exception as msg:
      print('错误内容=='+str(msg))
    finally:
      imageProcess.clean()
  return 'Success'

def enqueueOutput(out, queue, t):
  for line in iter(out.readline, b''):
    queue.put((t, line))
  out.flush()
  out.close()

def createEnqueueThread(pipe, t):
  t = threading.Thread(target=enqueueOutput, args=(pipe, qOut, t))
  t.daemon = True # thread dies with the program
  t.start()

def readSubprocess(q):
  while True:
    try:
      t, line = q.get_nowait()
      line = str(line, encoding='utf_8')
    except Empty:
      break
    else:
      if t == 0:
        sys.stdout.write(line)
      else:
        sys.stderr.write(line)

def SR_vid(video, scale=2, mode='a', dn_model='no', dnseq='', codec=defaultCodec):
  width, height, frameRate = getVideoInfo(video)
  video_out, videoName = Config().getPath()
  if not os.path.exists(video_out):
    os.mkdir(video_out)
  outputPath = video_out + '/' + videoName
  commandIn = [
    ffmpegPath,
    '-i', video,
    '-an',
    '-sn',
    '-f', 'rawvideo',
    '-s', '{}x{}'.format(width, height),
    '-pix_fmt', 'bgr48le',
    '-']
  commandOut = [
    ffmpegPath,
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr48le',
    '-s', '{}x{}'.format(width * scale, height * scale),
    '-r', str(frameRate),
    '-i', '-',
    '-i', video,
    '-map', '0:v',
    '-map', '1?',
    '-map', '-1:v',
    '-c:1', 'copy',
    '-c:v:0']
  commandOut.extend(codec.split(' '))
  commandOut.append(outputPath)
  print(video, scale, mode, dn_model, dnseq, commandOut)
  process = imageProcess.genProcess(scale, mode, dn_model, dnseq, 'buffer', 16)
  pipeIn = sp.Popen(commandIn, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)
  pipeOut = sp.Popen(commandOut, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8, shell=True)
  try:
    createEnqueueThread(pipeOut.stdout, 0)
    createEnqueueThread(pipeIn.stderr, 1)
    createEnqueueThread(pipeOut.stderr, 1)

    i = 0
    while True:
      raw_image = pipeIn.stdout.read(width * height * 6) # read width*height*6 bytes (= 1 frame)
      if len(raw_image) == 0:
        break
      i += 1
      readSubprocess(qOut)
      print('processing frame #{}'.format(i))
      buffer = process((raw_image, height, width))
      pipeOut.stdin.write(buffer)

    flag = threading.Event()
    flag.wait(0.1)
    pipeIn.stdout.flush()
    pipeOut.communicate()
  finally:
    pipeIn.terminate()
    pipeOut.kill()
    os.remove(video)
  readSubprocess(qOut)
  return outputPath

if __name__ == '__main__':
  print('video')
  SR_vid('./ves.mp4')