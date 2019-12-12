from shutil import copyfile
from gevent.event import Event
from config import config
config.videoPreview = ''
config.progressDetail = -1
from worker import context
from video import SR_vid

context.stopFlag = Event()
context.shared = None
video = 'upload\\realshort.mp4'
copyfile('test\\t2.mp4', video)
steps = [{'op': 'decode'}, {'op': 'range'}, {'codec': 'libx264 -pix_fmt yuv420p', 'op': 'encode', 'file': 'download/realshort.ts'}]
print(SR_vid(video, False, *steps))