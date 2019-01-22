import sys
from cx_Freeze import setup, Executable
import os
os.environ['TCL_LIBRARY'] = 'C:\\Users\\opteroncx.000\\Py36\\tcl\\tcl8.6'
os.environ['TK_LIBRARY'] = 'C:\\Users\\opteroncx.000\\Py36\\tcl\\tk8.6'
# Dependencies are automatically detected, but it might need fine tuning.

# include files
ifiles = ['./models.pyc',
        './turbo.pyc',
        './dehaze.pyc',
        './model.pyc',
        './runDN.pyc',
        './runSR.pyc',
        './video.pyc',
        './templates',
        './static',
        './model',
        './ffmpeg',
        './mkl_intel_thread.dll',
        './libiomp5md.dll',
        './libiomp5md.pdb',
        './libiompstubs5md.dll',
        './update_log.txt',
        './nvidia-smi.exe'
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
        'packages': ['tkinter', 'scipy', 'asyncio','numpy', 'torch', 'gevent'], 
        'includes': ['numpy.core._methods','jinja2','jinja2.ext','asyncio.compat'],
        'include_files': ifiles,
        'bin_excludes': efiles}

base = None

# if sys.platform == "win32":
#     base = "Win32GUI"

exe = Executable(script='MoePhoto.py', base = base, icon='logo3.ico')



setup(  name = 'MoePhoto',
        version = '2.5',
        description = 'May-workshop',
        options = {'build_exe': build_exe_options},
        executables = [exe])