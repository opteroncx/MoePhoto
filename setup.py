import sys
from cx_Freeze import setup, Executable
import os
import json
os.environ['TCL_LIBRARY'] = 'C:\\Users\\opteroncx.000\\Py36\\tcl\\tcl8.6'
os.environ['TK_LIBRARY'] = 'C:\\Users\\opteroncx.000\\Py36\\tcl\\tk8.6'
# Dependencies are automatically detected, but it might need fine tuning.

# include files
ifiles = [
        './pyc/models.pyc',
        './pyc/dehaze.pyc',
        './pyc/config.pyc',
        './pyc/imageProcess.pyc',
        './pyc/progress.pyc',
        './pyc/readgpu.pyc',
        './pyc/server.pyc',
        './pyc/worker.pyc',
        './pyc/runDN.pyc',
        './pyc/runSR.pyc',
        './pyc/video.pyc',
        './pyc/gan.pyc',
        './pyc/moe_utils.pyc',
        './pyc/mt_download.pyc',
        './pyc/slomo.pyc',
        './pyc/runSlomo.pyc',
        './pyc/MoeNet_lite2.pyc',
        './pyc/logger.pyc',
        './pyc/FIFOcache.pyc',
        './pyc/preset.pyc',
        './pyc/userConfig.pyc',
        './pyc/defaultConfig.pyc'
        './templates',
        './static',
        './model',
        './ffmpeg',
        './mkl_intel_thread.dll',
        './libiomp5md.dll',
        './libiomp5md.pdb',
        './libiompstubs5md.dll',
        './update_log.txt',
        './site-packages',
        './server.bat'
        ]

# exclude files
efiles = ['./model/__pycache__',
        './model/old',
        './model/convert_cpkt.py',
        './model/dn.py',
        './model/model.py',
        './model/model6.py',
        './model/model63.py',
        './model/model64.py',
        './model/model6dn.py'
        ]

build_exe_options = {
        'packages': ['tkinter', 'asyncio', 'numpy', 'torch', 'gevent','flask','torchvision','logging'],
        'includes': ['numpy.core._methods','jinja2','jinja2.ext','asyncio.compat'],
        'include_files': ifiles,
        'bin_excludes': efiles}

base = None

# if sys.platform == "win32":
#     base = "Win32GUI"

exe = Executable(script='./python/MoePhoto.py', base = base, icon='logo3.ico')


with open('package.json','r',encoding='utf-8') as manifest:
  version = json.load(manifest)['version']

setup(  name = 'MoePhoto',
        version = version,
        description = 'May-workshop',
        options = {'build_exe': build_exe_options},
        executables = [exe])