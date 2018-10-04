import os
import runSR
import runDN
import cv2
import shutil
from config import Config

def video2frame(video_path, frame_path, out_path):
    '''
    Split video frames
    '''
    if os.path.exists(out_path + '/audio.wav'):
        os.remove(out_path + '/audio.wav')
    os.system('.\\ffmpeg\\bin\\ffmpeg -i ' + video_path + ' ' + frame_path + '/f%06d.bmp')
    os.system('.\\ffmpeg\\bin\\ffmpeg -i ' + video_path + ' -vn -sn -acodec copy ' + out_path + '/audio.wav')

def frame2video(frame_path, out_path):
    os.system('.\\ffmpeg\\bin\\ffmpeg -y -r 24 -i ' + frame_path + '/f%06d.bmp -i -acodec copy ' + out_path + '/audio.wav ' + out_path + '/out.mkv')

def creat_tmp(tmp_folder):
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

def batchSR(path, srpath, scale, mode, dnmodel, dnseq):
    images = os.listdir(path)
    count = 0
    SRopt = runSR.getOpt(scale, mode)
    DNopt = runDN.getOpt(dnmodel)
    for image in images:
        count += 1
        print('processing frame #' + str(count))
        im_bgr = cv2.imread(path + '/' + image)
        im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        if (dnseq == 'before') and (dnmodel != 'no'):
            im = runDN.dn(im, DNopt)
        im = runSR.sr(im, SRopt)
        if (dnseq == 'after') and (dnmodel != 'no'):
            im = runDN.dn(im, DNopt)
        outputImg = im[..., ::-1]
        cv2.imwrite(srpath + '/' + image, outputImg)

def SR_vid(video, scale, mode, dn_model, dnseq):
    video_path = video
    frame_folder, srframe_folder, video_out = Config().getPath()
    creat_tmp(frame_folder)
    creat_tmp(srframe_folder)
    scale = scale
    mode = mode
    video2frame(video_path, frame_folder, video_out)
    batchSR(frame_folder, srframe_folder, scale, mode, dn_model, dnseq)
    frame2video(srframe_folder, video_out)
    copy2downloader()

def copy2downloader():
    if not os.path.exists('download'):
        os.mkdir('download')
    shutil.copyfile(out_path + '/out.mkv', 'download/output.mkv')

if __name__ == '__main__':
    print('video')
    SR_vid('./ves.mp4')