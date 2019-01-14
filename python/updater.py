# updater scripts

import os
import zipfile
import requests
import codecs
import shutil

releases = 'http://www.may-workshop.com/moephoto/version.html'
ufile = 'http://www.may-workshop.com/moephoto/files/'


def copyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile)) 

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
                copyfile('./update_tmp/'+f,'./python/update_tmp/'+f)
        print('升级完成')
        
        

if __name__ == '__main__':
    v = getVersion()
    print(v)
    update()