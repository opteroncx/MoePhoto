import os
from python import moe_utils
import shutil

# recompile files
os.system('npm install --no-save --no-audit')
os.system('npm update --no-save')
os.system('npm run build')
moe_utils.compile_pyc()
# os.system('python setup.py build')
# print('### fix scipy ####')
# os.rename('./build/exe.win-amd64-3.7/lib/scipy/spatial/cKDTree.cp37-win_amd64.pyd','./build/exe.win-amd64-3.7/lib/scipy/spatial/ckdtree.cp37-win_amd64.pyd')

# delete deprecated files
try:
    shutil.rmtree('../build/model')
    shutil.rmtree('../build/pyc')
    shutil.rmtree('../build/site-packages')
    shutil.rmtree('../build/ffmpeg')
    shutil.rmtree('../build/static')
    shutil.rmtree('../build/templates')
    shutil.rmtree('../build/download')
    shutil.rmtree('../build/.user')
    os.remove('../build/manifest.json')
    os.remove('../build/update_log.txt')
except:
    print('dir not exist')
# copy files
shutil.copytree(src='./model',dst='../build/model')
shutil.copytree(src='./pyc',dst='../build/pyc')
shutil.copytree(src='./site-packages',dst='../build/site-packages')
shutil.copytree(src='./ffmpeg',dst='../build/ffmpeg')
shutil.copytree(src='./static',dst='../build/static')
shutil.copytree(src='./templates',dst='../build/templates')
shutil.copy(src='./manifest.json',dst='../build/manifest.json')
shutil.copy(src='./update_log.txt',dst='../build/update_log.txt')