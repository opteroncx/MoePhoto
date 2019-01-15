# updater scripts

import os
import zipfile
import requests
import codecs
import shutil
from moe_utils import compile_pyc
from moe_utils import copyfile

releases = 'http://www.may-workshop.com/moephoto/version.html'
ufile = 'http://www.may-workshop.com/moephoto/files/'
ffmpeg = 'http://www.may-workshop.com/moephoto/files/ffmpeg.zip'
ff = 'https://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-20190114-d52a1be-win64-static.zip'
def update_model():
    print('更新模型文件')

def update_ffmpeg():
    print('更新FFMPEG')
    # first time, check path
    ffmpeg_home = './ffmpeg/bin/'
    if not os.path.exists(ffmpeg_home):
        os.makedirs(ffmpeg_home)
    # download files
    url = ff
    print('downloading from ',url)
    r = requests.get(url) 
    with open(ffmpeg_home+"ffmpeg.zip", "wb") as code:
        code.write(r.content)
    # extract zip
    z = zipfile.ZipFile(ffmpeg_home+'ffmpeg.zip', 'r')
    z.extractall(path=ffmpeg_home)
    z.close()

def getVersion(releases=releases):
    f = requests.get(releases)
    fv = f.text
    return fv[8:]

def update():
    # make temp dir
    if not os.path.exists('./update_tmp'):
        os.mkdir('./update_tmp')
    v = getVersion()
    log_file = codecs.open('./update_log.txt','r')
    current_v = log_file.readline()
    # version:xxx
    current_v = current_v[8:]
    print('current version==',current_v)
    if v<current_v:
        print('已是最新版本')
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
        
        

if __name__ == '__main__':
    v = getVersion()
    print(v)
    update_ffmpeg()