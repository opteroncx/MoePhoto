import os
from runSR import dosr
from runDN import dodn
import cv2
import shutil

def video2frame(video_path, out_path):
    '''
    Split video frames
    '''
    if os.path.exists('temp/audio.mp3'):
        os.remove('temp/audio.mp3')
    os.system('.\\ffmpeg\\bin\\ffmpeg -i ' + video_path + ' temp/video_frame/frames_%05d.png')
    os.system('.\\ffmpeg\\bin\\ffmpeg -i ' + video_path + ' temp/audio.mp3')


def frame2video(frame_path, out_path):
    os.system('.\\ffmpeg\\bin\\ffmpeg -threads 2 -y -r 24 -i ' + frame_path + '/frames_%05d.png -i temp/audio.mp3 ' + out_path + '/out.mp4')


def creat_tmp(tmp_folder):
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)


def batchSR(path, srpath, scale, mode,dnmodel, dnseq):
    images = os.listdir(path)
    for image in images:
        im_bgr = cv2.imread(path + '/' + image)
        im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        if dnseq == 'before':
            if dnmodel == 'no':
                outputImg = dosr(im, scale, mode)
            else:
                dim = dodn(im, dnmodel)
                outputImg = dosr(im, scale, mode)
        elif dnseq == 'after':
            if dnmodel == 'no':
                outputImg=dosr(im, scale, mode)
            else:
                sim = dosr(im, scale, mode)
                outputImg = dodn(sim, dnmodel)
        outputImg = outputImg[..., ::-1]
        cv2.imwrite(srpath + '/' + image, outputImg)
    


def SR_vid(video, scale, mode, dn_model, dnseq):
    video_path = video
    frame_folder = './temp/video_frame'
    srframe_folder = './temp/video_frame_sr'
    video_out = './temp'
    creat_tmp(frame_folder)
    creat_tmp(srframe_folder)
    scale = scale
    mode = mode
    video2frame(video_path, frame_folder)
    batchSR(frame_folder, srframe_folder, scale, mode,dn_model,dnseq)
    frame2video(srframe_folder, video_out)
    copy2downloader()


def copy2downloader():
    if not os.path.exists('download'):
        os.mkdir('download')
    shutil.copyfile('temp/out.mp4', 'download/output.mp4')

if __name__ == '__main__':
    print('video')
    SR_vid('./ves.mp4')