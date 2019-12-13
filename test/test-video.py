# pylint: disable=E0401
from shutil import copyfile
from gevent.event import Event
import sys
sys.path.append('./python')
from config import config
config.videoPreview = ''
config.progressDetail = -1
from worker import context
from video import SR_vid

context.stopFlag = Event()
context.shared = None
video = 'upload\\realshort.mp4'
copyfile('test\\realshort.mp4', video)
steps = [{'op': 'decode'}, {'op': 'range'}, {'codec': 'copy', 'op': 'encode', 'file': 'download/realshort.ts'}]
print(SR_vid(video, False, *steps))