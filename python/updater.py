# updater scripts

import os
import zipfile
import requests
import codecs
releases = 'http://www.may-workshop.com/moephoto/release.txt'
ufile = 'http://www.may-workshop.com/moephoto/files/'


def getVersion():
    f = open()
    fv = f.readlines[-1]
    return fv

def update():
    # make temp dir
    if not os.path.exists('./update_tmp'):
        os.mkdir('./update_tmp')
    v = getVersion()
    log_file = codecs.open('./update_log.txt','utf-8')
    current_v = log_file.readline()
    # version:xxx
    current_v = current_v[7:]
    if v<current_v:
        print('已是最新版本')
    else:
        url_new_version = ufile+v
        # download zip
        print('downloading')
        url = url_new_version 
        r = requests.get(url) 
        with open("./update_tmp/tmp.zip", "wb") as code:
            code.write(r.content)
        # extract zip
        z = zipfile.ZipFile('./update_tmp/tmp.zip', 'r')
        z.extractall(path='./update_tmp')
        z.close()
        print('升级完成')
