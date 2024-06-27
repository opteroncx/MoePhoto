import os
import os.path as osp
import json
from python import moe_utils
from python.updater import update_ffmpeg, isWindows
import shutil
import argparse
manifestPath = './manifest.json'

parser = argparse.ArgumentParser(description='Moe build')
parser.add_argument('--npm', default=True, nargs='?', const=True, type=eval,
                    help='install and update npm packages (default: %(default)s)')
parser.add_argument('--compile', default=True, nargs='?', const=True, type=eval,
                    help='recompile files (default: %(default)s)')
parser.add_argument('--build_ext', default=True, nargs='?', const=True, type=eval,
                    help='build CUDA extensions (default: %(default)s)')
parser.add_argument('--clean', default=True, nargs='?', const=True, type=eval,
                    help='clean old built files (default: %(default)s)')
parser.add_argument('--ffmpeg', default=False, nargs='?', const=True, type=eval,
                    help='download latest ffmpeg from Internet (default: %(default)s)')
parser.add_argument('--copy', default=True, nargs='?', const=True, type=eval,
                    help='copy built files to deploy folder (default: %(default)s)')
parser.add_argument('--platform', default='native', choices=['native', 'win', 'linux'],
                    help='destination platform to deploy (default: %(default)s)')

os.chdir(osp.dirname(osp.abspath(__file__)))
with open('package.json','r',encoding='utf-8') as manifest:
  version = json.load(manifest)['version'].split('-')[0]

parser.add_argument('--version', action='version', version=version)

args = parser.parse_args()

manifest = {
  'version': version,
  'releases': 'https://moephoto.tech/moephoto/version.html',
  'ufile': 'https://moephoto.tech/moephoto/files/',
  'ffmpeg-win': 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z',
  'ffmpeg-linux': 'https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz'
}

with open(manifestPath, 'w') as f:
  json.dump(manifest, f, ensure_ascii=False)

if args.npm:
  os.system('npm install --no-save --no-audit')
  os.system('npm update --no-save')

if args.compile:
  os.system('npm run build')
  moe_utils.compile_pyc()

if args.build_ext:
  os.system('python site-packages/ailut/setup.py build')
  folder = 'site-packages/ailut/build'
  name = next(name for name in os.listdir(folder) if name.startswith('lib.'))
  shutil.copytree(osp.join(folder, name, 'ailut'), 'python/ailut', dirs_exist_ok=True)
  shutil.rmtree(folder)

files = {
  'presets': '.user',
  'model': 'model',
  'python scripts': 'pyc',
  'site-packages': 'site-packages',
  'ffmpeg': 'ffmpeg',
  'static': 'static',
  'templates': 'templates',
  'manifest': 'manifest.json',
  'update_log': 'update_log.txt',
}
cleanFiles = { 'download': 'download' }
cleanFiles.update(files)

getBuild = lambda v: '../build/{}'.format(v)
getDev = lambda v: './{}'.format(v)

if args.clean:
  for key in cleanFiles:
    moe_utils.delete_files(getBuild(cleanFiles[key]))

if args.ffmpeg:
  platform = isWindows if args.platform == 'native' else (args.platform == 'win')
  try:
    update_ffmpeg(manifest, platform)
  except Exception:
    print('update ffmpeg failed')

if args.copy:
  for key in cleanFiles:
    print('copying {}'.format(key))
    v = cleanFiles[key]
    try:
      shutil.copytree(src=getDev(v),dst=getBuild(v))
    except Exception:
      shutil.copy(src=getDev(v),dst=getBuild(v))