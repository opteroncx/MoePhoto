# updater scripts

import os
import zipfile
import requests

releases = 'http://www.may-workshop.com/moephoto/release.txt'
ufile = 'http://www.may-workshop.com/moephoto/files/'

def getVersion():
    f = open()
    fv = f.readlines[-1]
    return fv

def update():
    v = getVersion()
    new = ufile+v
    # download zip
    if not os.path.exists('./update_tmp'):
        os.mkdir('./update_tmp')

    # extract zip
