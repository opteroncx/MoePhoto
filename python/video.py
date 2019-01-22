import os
import subprocess as sp
import re
import sys
import threading
from io import BytesIO
from queue import Queue, Empty
from gevent import spawn_later
from config import config
from imageProcess import genProcess, clean, writeFile, BGR2RGB
from progress import Node, initialETA

ffmpegPath = os.path.realpath('ffmpeg/bin/ffmpeg') # require full path to spawn in shell
qOut = Queue(64)

def getVideoInfo(videoPath):
  commandIn = [
    ffmpegPath,
    '-i', videoPath,
    '-map', '0:v:0',
    '-c', 'copy',
    '-f', 'null',
    '-'
  ]
  try:
    pipeIn = sp.Popen(commandIn, stderr=sp.PIPE, encoding='utf_8')
    totalFrames = 0

    for line in iter(pipeIn.stderr.readline, ''):
      line = line.lstrip()
      if re.match('Stream #.*: Video:', line):
        try:
          videoInfo = re.search(',[\\s]*([\\d]+)x([\\d]+)[\\s]*.+,[\\s]*([.\\d]+)[\\s]*fps', line).groups()
          width = int(videoInfo[0])
          height = int(videoInfo[1])
          frameRate = float(videoInfo[2])
        except:
          print(line)
          raise RuntimeError('Video info not found')
      if re.match('frame=', line):
        try:
          totalFrames = int(re.search('frame=[\\s]*([\\d]+) ', line).groups()[0])
        except:
          print(line)

    pipeIn.stderr.flush()
    pipeIn.stderr.close()
  finally:
    pipeIn.terminate()
  print('Info of video {}: {}x{}@{}fps, {} frames'.format(videoPath, width, height, frameRate, totalFrames))
  return width, height, frameRate, totalFrames

def batchSR(images, srpath, scale, mode, dnmodel, dnseq, begin=None):
  count = 0
  fail = 0
  result = 'Success'
  process, nodes = genProcess(scale, mode, dnmodel, dnseq, 'file')
  root, _ = begin(Node({'op': 'batch'}, 1, len(images), 0), nodes, False)
  root.reset().trace(0)
  for image in images:
    if not root.running:
      result = 'Interrupted'
    count += 1
    print('processing image {}'.format(image.filename))
    fileName = '{}{}.png'.format(srpath, count)
    try:
      process(image, fileName)
    except Exception as msg:
      print('错误内容=='+str(msg))
    finally:
      clean()
      fail += 1
    root.trace(preview=fileName)
  return result, count, fail

def enqueueOutput(out, queue, t):
  for line in iter(out.readline, b''):
    queue.put((t, line))
  out.flush()

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

def SR_vid(video, begin, opt={}):
  scale = int(opt['scale']) if 'scale' in opt else 2
  mode = opt['mode'] if 'mode' in opt else 'a'
  dn_model = opt['dn_model'] if 'dn_model' in opt else 'no'
  dnseq = opt['dnseq'] if 'dnseq' in opt else ''
  codec = opt['codec'] if 'codec' in opt else config.defaultCodec  # pylint: disable=E1101
  start = int(opt['start']) if 'start' in opt else 1
  outDir = opt['outDir'] if 'outDir' in opt else ''
  process, nodes = genProcess(scale, mode, dn_model, dnseq, 'buffer', 16)
  root, current = begin(Node({'op': 'video', 'codec': codec}, 1, 2, 0), nodes, False)
  width, height, frameRate, totalFrames = getVideoInfo(video)
  if start > totalFrames:
    return ''
  if start < 1:
    start = 1
  stop = int(opt['stop']) if 'stop' in opt else totalFrames
  if stop < start:
    stop = totalFrames
  elif stop > totalFrames:
    stop = totalFrames
  root.total = stop - start + 1
  for n in nodes:
    n.load *= width * height * 3
  initialETA(root)
  root.reset().trace(0)
  videoName = config.getPath()
  outputPath = outDir + '/' + videoName
  previewPath = outDir + '/preview.png'
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
  if start > 1:
    commandOut = commandOut[:12] + commandOut[22:]
  commandOut.extend(codec.split(' '))
  commandOut.append(outputPath)
  pipeIn = sp.Popen(commandIn, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)
  pipeOut = sp.Popen(commandOut, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8, shell=True)
  preview = BytesIO()
  try:
    createEnqueueThread(pipeOut.stdout, 0)
    createEnqueueThread(pipeIn.stderr, 1)
    createEnqueueThread(pipeOut.stderr, 1)

    i = 0
    while root.running and i <= stop:
      raw_image = pipeIn.stdout.read(width * height * 6) # read width*height*6 bytes (= 1 frame)
      if len(raw_image) == 0:
        break
      i += 1
      readSubprocess(qOut)
      if i < start:
        continue
      buffer, image = process((raw_image, height, width))
      pipeOut.stdin.write(buffer)
      preview.seek(0)
      writeFile(BGR2RGB(image), preview, 'PNG')
      preview.seek(0)
      current.previewIm = preview.getbuffer().tobytes()
      root.trace(preview=previewPath)

    pipeIn.stderr.flush()
    pipeOut.stdout.flush()
    pipeOut.stderr.flush()
    pipeOut.communicate()
  finally:
    pipeIn.terminate()
    pipeOut.terminate()
    clean()
    spawn_later(5, remove, video)
  readSubprocess(qOut)
  return outputPath, i

def remove(path):
  try:
    os.remove(path)
  except:
    print('Timed out waiting ffmpeg to terminate, need to remove {} manually.'.format(path))

if __name__ == '__main__':
  print('video')
  SR_vid('./ves.mp4', lambda x, _: (x, lambda:None))