import os
import subprocess as sp
import re
import sys
import threading
import logging
from queue import Queue, Empty
from gevent import spawn_later, idle
from config import config
from imageProcess import clean, writeFile, BGR2RGB
from procedure import genProcess
from progress import Node, initialETA
from worker import context, begin

log = logging.getLogger('Moe')
ffmpegPath = os.path.realpath('ffmpeg/bin/ffmpeg') # require full path to spawn in shell
qOut = Queue(256)
console = [sys.stdout, sys.stderr]
stepVideo = [dict(op='buffer', bitDepth=16)]
pix_fmt = 'bgr48le'
pixBytes = 6
bufsize = 10 ** 8
reMatchInfo = re.compile(r'Stream #.*: Video:')
reSearchInfo = re.compile(r',[\s]*([\d]+)x([\d]+)[\s]*.+,[\s]*([.\d]+)[\s]*fps')
reMatchFrame = re.compile(r'frame=')
reSearchFrame = re.compile(r'frame=[\s]*([\d]+) ')
popen = lambda command: sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
popenText = lambda command: sp.Popen(command, stderr=sp.PIPE, encoding='utf_8', errors='ignore')
insert1 = lambda t, s: ''.join((t[0], s, *t[1:]))
suffix = lambda p, s: insert1(os.path.splitext(p), s)

def removeFile(path):
  try:
    os.remove(path)
  except FileNotFoundError: pass

def getVideoInfo(videoPath, procIn, width, height, frameRate):
  commandIn = [
    ffmpegPath,
    '-hide_banner',
    '-i', videoPath,
    '-map', '0:v:0',
    '-c', 'copy',
    '-f', 'null',
    '-'
  ]
  matchInfo = not (width and height and frameRate)
  matchFrame = not bool(procIn)
  error = RuntimeError('Video info not found')
  try:
    if matchFrame:
      procIn = popenText(commandIn)
    totalFrames = 0

    while matchInfo or matchFrame:
      line = procIn.stderr.readline()
      if type(line) != str:
        line = str(line, 'utf-8', errors='ignore')
      sys.stdout.write(line)
      if not line:
        break
      line = line.lstrip()
      if matchInfo and reMatchInfo.match(line):
        try:
          videoInfo = reSearchInfo.search(line).groups()
          if not width:
            width = int(videoInfo[0])
          if not height:
            height = int(videoInfo[1])
          if not frameRate:
            frameRate = float(videoInfo[2])
        except:
          log.error(line)
          raise error
        matchInfo = False
      if matchFrame and reMatchFrame.match(line):
        try:
          totalFrames = int(reSearchFrame.search(line).groups()[0])
        except:
          log.error(line)

    if matchFrame:
      procIn.stderr.flush()
      procIn.stderr.close()
  finally:
    if matchFrame:
      procIn.terminate()
  if matchInfo or (matchFrame and not totalFrames):
    raise error
  log.info('Info of video {}: {}x{}@{}fps, {} frames'.format(videoPath, width, height, frameRate, totalFrames))
  return width, height, frameRate, totalFrames

def enqueueOutput(out, queue, t):
  try:
    for line in iter(out.readline, b''):
      queue.put((t, line))
    out.flush()
  except: pass

def createEnqueueThread(pipe, *args):
  t = threading.Thread(target=enqueueOutput, args=(pipe, qOut, *args))
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
      console[t].write(line)

def prepare(video, by, steps):
  optEncode = steps[-1]
  encodec = optEncode['codec'] if 'codec' in optEncode else config.defaultEncodec  # pylint: disable=E1101
  optDecode = steps[0]
  decodec = optDecode['codec'] if 'codec' in optDecode else config.defaultDecodec  # pylint: disable=E1101
  optRange = steps[1]
  start = int(optRange['start']) if 'start' in optRange else 0
  outDir = config.outDir  # pylint: disable=E1101
  procSteps = stepVideo + list(steps[2:-1])
  process, nodes = genProcess(procSteps)
  traceDetail = config.progressDetail  # pylint: disable=E1101
  root = begin(Node({'op': 'video', 'encodec': encodec}, 1, 2, 0), nodes, traceDetail)
  context.root = root
  slomos = [*filter((lambda opt: opt['op'] == 'slomo'), procSteps)]
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
      stop = -1
  else:
    stop = -1
  root.total = -1 if stop < 0 else stop - start
  outputPath = optEncode.get('file', '') or outDir + '/' + config.getPath()
  dataPath = suffix(outputPath, '-a')
  commandIn = [
    ffmpegPath,
    '-hide_banner',
    '-i', video,
    '-vn',
    '-c', 'copy',
    '-y',
    dataPath,
    '-map', '0:v',
    '-f', 'rawvideo',
    '-pix_fmt', pix_fmt]
  if len(decodec):
    commandIn.extend(decodec.split(' '))
  commandIn.append('-')
  metadata = ['-metadata', 'service_provider="MoePhoto {}"'.format(config.version)] # pylint: disable=E1101
  commandVideo = [
    ffmpegPath,
    '-hide_banner', '-y',
    '-f', 'rawvideo',
    '-pix_fmt', pix_fmt,
    '-s', '',
    '-r', '',
    '-i', '-',
    '-i', dataPath,
    '-map', '0:v',
    '-map', '1?',
    '-map', '-1:v',
    '-c:1', 'copy',
    *metadata,
    '-c:v:0'
  ] + encodec.split(' ') + [outputPath]
  if by:
    commandVideo[-1] = suffix(outputPath, '-v')
    commandVideo = commandVideo[:13] + commandVideo[23:]
    commandOut = [
      ffmpegPath,
      '-hide_banner', '-y',
      '-i', commandVideo[-1],
      '-i', dataPath,
      '-c:0', 'copy',
      '-c:1', 'copy',
      *metadata,
      outputPath
    ]
  else:
    commandIn = commandIn[:4] + commandIn[9:]
    commandVideo[14] = video
    if start > 0:
      commandVideo = commandVideo[:13] + commandVideo[23:]
    commandOut = None
  if start > 0:
    commandOut = None
    commandVideo[-1] = outputPath
  frameRate = optEncode.get('frameRate', 0)
  width = optDecode.get('width', 0)
  height = optDecode.get('height', 0)
  sizes = filter((lambda opt: opt['op'] == 'SR' or opt['op'] == 'resize'), procSteps)
  return outputPath, process, start, stop, root, commandIn, commandVideo, commandOut, slomos, sizes, width, height, frameRate

def setupInfo(root, commandVideo, slomos, sizes, start, width, height, frameRate, totalFrames):
  if root.total < 0 and totalFrames > 0:
    root.total = totalFrames - start
  if frameRate:
    for opt in slomos:
      frameRate *= opt['sf']
  outWidth, outHeight = (width, height)
  for opt in sizes:
    if opt['op'] == 'SR':
      outWidth *= opt['scale']
      outHeight *= opt['scale']
    else: # resize
      outWidth = round(outWidth * opt['scaleW']) if 'scaleW' in opt else opt['width']
      outHeight = round(outHeight * opt['scaleH']) if 'scaleH' in opt else opt['height']
  commandVideo[8] = f'{outWidth}x{outHeight}'
  commandVideo[10] = str(frameRate)
  root.multipleLoad(width * height * 3)
  initialETA(root)
  root.reset().trace(0)

def cleanAV(command):
  if command:
    removeFile(command[4])
    removeFile(command[6])

def mergeAV(command):
  if command:
    procMerge = popenText(command)
    createEnqueueThread(procMerge.stderr, 1)
    procMerge.communicate()
    return procMerge

def SR_vid(video, by, *steps):
  def p(raw_image=None):
    bufs = process((raw_image, height, width))
    if (not bufs is None) and len(bufs):
      for buffer in bufs:
        if buffer:
          procOut.stdin.write(buffer)
          procOut.stdin.flush()
    if raw_image:
      root.trace()

  outputPath, process, start, stop, root, commandIn, commandVideo, commandOut, slomos, sizes, *info = prepare(video, by, steps)
  if by:
    procIn = popen(commandIn)
    width, height, *more = getVideoInfo(video, procIn, *info)
  else:
    width, height, *more = getVideoInfo(video, False, *info)
    procIn = popen(commandIn)
  setupInfo(root, commandVideo, slomos, sizes, start, width, height, *more)
  procOut = sp.Popen(commandVideo, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize, shell=True)

  try:
    createEnqueueThread(procOut.stdout, 0)
    createEnqueueThread(procIn.stderr, 1)
    createEnqueueThread(procOut.stderr, 1)
    i = 0
    while (stop < 0 or i <= stop) and not context.stopFlag.is_set():
      print(i)
      raw_image = procIn.stdout.read(width * height * pixBytes) # read width*height*6 bytes (= 1 frame)
      if len(raw_image) == 0:
        break
      readSubprocess(qOut)
      if i >= start:
        p(raw_image)
      i += 1
      idle()
    procIn.terminate()
    p()

    procOut.communicate(timeout=300)
    readSubprocess(qOut)
    procMerge = mergeAV(commandOut)
  finally:
    procIn.terminate()
    procOut.terminate()
    procMerge.terminate()
    clean()
    try:
      if not by:
        removeFile(video)
      cleanAV(commandOut)
    except:
      log.warning('Timed out waiting ffmpeg to terminate, need to remove {} manually.'.format(video))
  readSubprocess(qOut)
  return outputPath, i