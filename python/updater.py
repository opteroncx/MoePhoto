# updater scripts

import os
import sys
import zipfile
import tarfile
import requests
import shutil
from moe_utils import compile_pyc
from moe_utils import copyfile
from mt_download import download_file
from defaultConfig import defaultConfig
from userConfig import compareVersion
releases = 'http://www.may-workshop.com/moephoto/version.html'
ufile = 'http://www.may-workshop.com/moephoto/files/'

ffmpeg_home = './ffmpeg/bin/'
isWindows = 1 if sys.platform[:3] == 'win' else 0
ff = 'https://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-latest-win64-static.zip' if isWindows else 'https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz'
ffname = 'ffmpeg.zip' if isWindows else 'ffmpeg.tar.xz'
outfname = 'ffmpeg.exe' if isWindows else 'ffmpeg'

def update_model():
    print('更新模型文件')

def update_ffmpeg():
    print('更新FFMPEG')
    # first time, check path
    if not os.path.exists(ffmpeg_home):
        os.makedirs(ffmpeg_home)
    # download files
    url = ff
    print('downloading from ',url)
    fname = '{}{}'.format(ffmpeg_home, ffname)
    outPath = '{}{}'.format(ffmpeg_home, outfname)
    download_file(ff,fname=fname)

    if isWindows:
        # extract zip
        ndir = 'ffmpeg-latest-win64-static'
        z = zipfile.ZipFile(fname, 'r')
        z.extract('{}/bin/ffmpeg.exe'.format(ndir), path=ffmpeg_home)
        z.close()
        copyfile('{}{}/bin/ffmpeg.exe'.format(ffmpeg_home, ndir),outPath)
        # clean tmp
        shutil.rmtree(ffmpeg_home+ndir)
    else:
        tar = tarfile.open(fname)
        file = tar.extractfile('{}/ffmpeg'.format(tar.getnames()[0]))
        buf = file.read()
        tar.close()
        with open(outPath, 'wb') as out:
            out.write(buf)
    os.remove(fname)

def getVersion(releases=releases):
    f = requests.get(releases)
    fv = f.text
    return fv[8:]

def update():
    # make temp dir
    if not os.path.exists('./update_tmp'):
        os.mkdir('./update_tmp')
    v = getVersion()
    current_v = defaultConfig['version'][0]
    print('current version==',current_v)
    if compareVersion(v, current_v) <= 0:
        print('已是最新版本')
        result = '已是最新版本'
    else:
        url_new_version = ufile+v+'.zip'
        # download zip
        print('downloading from ',url_new_version)
        url = url_new_version 
        r = requests.get(url) 
        with open("./update_tmp/tmp.zip", "wb") as code:
            code.write(r.content)
        # extract zip
        z = zipfile.ZipFile('./update_tmp/tmp.zip', 'r')
        z.extractall(path='./update_tmp')
        z.close()
        # copy files
        print('copying files')
        py_files = os.listdir('./update_tmp')
        for f in py_files:
            if f[-3:] == '.py':
                # copyfile('./update_tmp/'+f,'./python/update_tmp/'+f)
                compile_pyc(path='./update_tmp/')
                copyfile('./update_tmp/__pycache__/'+f[:-3]+'.pyc','./'+f[:-3]+'.pyc')
                # recompile pyc
            elif f[-4:] == '.txt':
                copyfile('./update_tmp/'+f,'./'+f)
        print('升级完成,请重启软件')        
        #clean temp files
        shutil.rmtree('./update_tmp')
        result = '升级完成,请重启软件'
    return result

if __name__ == '__main__':
    v = getVersion()
    print(v)
    update_ffmpeg()