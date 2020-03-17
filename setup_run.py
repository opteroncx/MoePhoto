import os
import json
from python import moe_utils
from python.updater import update_ffmpeg
import shutil
manifestPath = './manifest.json'

# recompile files
os.system('npm install --no-save --no-audit')
os.system('npm update --no-save')
os.system('npm run build')
moe_utils.compile_pyc()
# os.system('python setup.py build')
# print('### fix scipy ####')
# os.rename('./build/exe.win-amd64-3.7/lib/scipy/spatial/cKDTree.cp37-win_amd64.pyd','./build/exe.win-amd64-3.7/lib/scipy/spatial/ckdtree.cp37-win_amd64.pyd')

# delete deprecated files

with open('package.json','r',encoding='utf-8') as manifest:
  version = json.load(manifest)['version'].split('-')[0]

manifest = {
  'version': version,
  'releases': 'https://moephoto.tech/moephoto/version.html',
  'ufile': 'https://moephoto.tech/moephoto/files/',
  'ffmpeg-win': 'https://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-latest-win64-static.zip',
  'ffmpeg-linux': 'https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz'
}

with open(manifestPath, 'w') as f:
  json.dump(manifest, f, ensure_ascii=False)

moe_utils.delete_files('../build/model')
moe_utils.delete_files('../build/pyc')
moe_utils.delete_files('../build/site-packages')
moe_utils.delete_files('../build/ffmpeg')
moe_utils.delete_files('../build/static')
moe_utils.delete_files('../build/templates')
moe_utils.delete_files('../build/manifest.json')
moe_utils.delete_files('../build/update_log.txt')
moe_utils.delete_files('../build/download')
moe_utils.delete_files('../build/.user')
update_ffmpeg(manifest)
# copy files
print('copying presets')
shutil.copytree(src='./.user',dst='../build/.user')
print('copying model')
shutil.copytree(src='./model',dst='../build/model')
print('copying python scripts')
shutil.copytree(src='./pyc',dst='../build/pyc')
print('copying site-packages')
shutil.copytree(src='./site-packages',dst='../build/site-packages')
print('copying ffmpeg')
shutil.copytree(src='./ffmpeg',dst='../build/ffmpeg')
print('copying static')
shutil.copytree(src='./static',dst='../build/static')
print('copying templates')
shutil.copytree(src='./templates',dst='../build/templates')
print('copying manifest')
shutil.copy(src='./manifest.json',dst='../build/manifest.json')
print('copying update_log')
shutil.copy(src='./update_log.txt',dst='../build/update_log.txt')