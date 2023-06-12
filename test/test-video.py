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
video = 'upload/t1.mkv'
copyfile('F:/animation/t1.mkv', video)
op2 = {'op': 'slomo', 'sf': 2.0, 'model': 'M'}
steps = [{'op': 'decode'}, {'op': 'range'}, op2, {'codec': 'hevc_nvenc -rc vbr -cq 26 -preset:v p7 -profile:v main -tune hq -tier high -bf 5 -b_ref_mode middle -spatial_aq 1 -temporal-aq 1 -nonref_p 1 -rc-lookahead 53 -keyint_min 1 -pix_fmt p010le', 'op': 'encode', 'file': 'download/t1.mkv'}]
print(SR_vid(video, False, *steps))