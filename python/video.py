import os
import subprocess as sp
import re
import sys
import threading
import logging
from queue import Queue, Empty
from gevent import spawn_later, idle
from config import config
from imageProcess import genProcess, clean, writeFile, BGR2RGB
from progress import Node, initialETA
from worker import context, begin

log = logging.getLogger('Moe')
ffmpegPath = os.path.realpath('ffmpeg/bin/ffmpeg') # require full path to spawn in shell
qOut = Queue(64)
stepVideo = [dict(op='buffer', bitDepth=16)]
pix_fmt = 'bgr48le'
pixBytes = 6
bufsize = 10 ** 8

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
    pipeIn = sp.Popen(commandIn, stderr=sp.PIPE, encoding='utf_8', errors='ignore')
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
          log.error(line)
          raise RuntimeError('Video info not found')
      if re.match('frame=', line):
        try:
          totalFrames = int(re.search('frame=[\\s]*([\\d]+) ', line).groups()[0])
        except:
          log.error(line)

    pipeIn.stderr.flush()
    pipeIn.stderr.close()
  finally:
    pipeIn.terminate()
  log.info('Info of video {}: {}x{}@{}fps, {} frames'.format(videoPath, width, height, frameRate, totalFrames))
  return width, height, frameRate, totalFrames

def enqueueOutput(out, queue, t):
  try:
    for line in iter(out.readline, b''):
      queue.put((t, line))
    out.flush()
  except: pass

def createEnqueueThread(pipe, t):
  t = threading.Thread(target=enqueueOutput, args=(pipe, qOut, t))
  t.daemon = True # thread dies with the program
  t.start()

def readSubprocess(q):
  while True:
    try:
      t, line = q.get_nowait()
      line = str(line, encoding='utf_8', errors='replace')
    except Empty:
      break
    else:
      if t == 0:
        sys.stdout.write(line)
      else:
        sys.stderr.write(line)

def prepare(video, steps):
  optEncode = steps[-1]
  encodec = optEncode['codec'] if 'codec' in optEncode else config.defaultEncodec  # pylint: disable=E1101
  optDecode = steps[0]
  decodec = optDecode['codec'] if 'codec' in optDecode else config.defaultDecodec  # pylint: disable=E1101
  optRange = steps[1]
  start = int(optRange['start']) if 'start' in optRange else 0
  outDir = config.outDir  # pylint: disable=E1101
  procSteps = stepVideo + list(steps[2:-1])
  process, nodes = genProcess(procSteps)
  root = begin(Node({'op': 'video', 'encodec': encodec}, 1, 2, 0), nodes, False)
  context.root = root
  width, height, frameRate, totalFrames = getVideoInfo(video)
  slomos = [*filter((lambda opt: opt['op'] == 'slomo'), procSteps)]
  if 'frameRate' in optEncode:
    frameRate = optEncode['frameRate']
  else:
    for opt in slomos:
      frameRate *= opt['sf']
  if 'width' in optDecode:
    width = optDecode['width']
  if 'height' in optDecode:
    height = optDecode['height']
  outWidth, outHeight = (width, height)
  for opt in filter((lambda opt: opt['op'] == 'SR' or opt['op'] == 'resize'), procSteps):
    if opt['op'] == 'SR':
      outWidth *= opt['scale']
      outHeight *= opt['scale']
    else: # resize
      outWidth = round(outWidth * opt['scaleW']) if 'scaleW' in opt else opt['width']
      outHeight = round(outHeight * opt['scaleH']) if 'scaleH' in opt else opt['height']
  if start < 0:
    start = 0
  if start and len(slomos): # should generate intermediate frames between start-1 and start
    start -= 1
    for opt in slomos:
      opt['opt'].firstTime = 0
  stop = None
  if 'stop' in optRange:
    stop = int(optRange['stop'])
    if stop <= start:
      stop = None
  root.total = (stop if stop else totalFrames) - start
  if not stop:
    stop = 0xffffffff
  root.multipleLoad(width * height * 3)
  initialETA(root)
  root.reset().trace(0)
  videoName = config.getPath()
  outputPath = outDir + '/' + videoName
  commandIn = [
    ffmpegPath,
    '-i', video,
    '-an',
    '-sn',
    '-f', 'rawvideo',
    '-s', '{}x{}'.format(width, height),
    '-pix_fmt', pix_fmt]
  if len(decodec):
    commandIn.extend(decodec.split(' '))
  commandIn.append('-')
  commandOut = [
    ffmpegPath,
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', pix_fmt,
    '-s', '{}x{}'.format(outWidth, outHeight),
    '-r', str(frameRate),
    '-i', '-',
    '-i', video,
    '-map', '0:v',
    '-map', '1?',
    '-map', '-1:v',
    '-c:1', 'copy',
    '-c:v:0']
  if start > 0:
    commandOut = commandOut[:12] + commandOut[22:]
  if len(encodec):
    commandOut.extend(encodec.split(' '))
  commandOut.append(outputPath)
  return commandIn, commandOut, outputPath, width, height, start, stop, root, process

def SR_vid(video, *steps):
  commandIn, commandOut, outputPath, width, height, start, stop, root, process = prepare(video, steps)
  pipeIn = sp.Popen(commandIn, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
  pipeOut = sp.Popen(commandOut, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize, shell=True)
  def p(raw_image=None):
    bufs = process((raw_image, height, width))
    if type(bufs) != type(None) and len(bufs):
      for buffer in bufs:
        if buffer:
          pipeOut.stdin.write(buffer)
    if raw_image:
      root.trace()

  try:
    createEnqueueThread(pipeOut.stdout, 0)
    createEnqueueThread(pipeIn.stderr, 1)
    createEnqueueThread(pipeOut.stderr, 1)
    i = 0
    while i <= stop and not context.stopFlag.is_set():
      raw_image = pipeIn.stdout.read(width * height * pixBytes) # read width*height*6 bytes (= 1 frame)
      if len(raw_image) == 0:
        break
      readSubprocess(qOut)
      if i >= start:
        p(raw_image)
      i += 1
      idle()
    p()

    pipeOut.communicate()
  finally:
    pipeIn.terminate()
    pipeOut.terminate()
    clean()
    try:
      os.remove(video)
    except:
      log.warning('Timed out waiting ffmpeg to terminate, need to remove {} manually.'.format(video))
  readSubprocess(qOut)
  return outputPath, i