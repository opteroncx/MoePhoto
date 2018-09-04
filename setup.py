import sys
from cx_Freeze import setup, Executable
import os
os.environ['TCL_LIBRARY'] = 'C:\\Users\\opteroncx.000\\Py36\\tcl\\tcl8.6'
os.environ['TK_LIBRARY'] = 'C:\\Users\\opteroncx.000\\Py36\\tcl\\tk8.6'
# Dependencies are automatically detected, but it might need fine tuning.

ifiles = ['./models.pyc',
        './model6dn.pyc',
        './dn.pyc',
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

build_exe_options = {
        'packages': ['tkinter', 'scipy', 'asyncio','numpy','torch'], 
        'includes': ['numpy.core._methods','jinja2','jinja2.ext','asyncio.compat'],
        'include_files': ifiles}

base = None

# if sys.platform == "win32":
#     base = "Win32GUI"

exe = Executable(script='MoePhoto.py', base = base, icon='logo3.ico')



setup(  name = 'MoePhoto',
        version = '1.5',
        description = 'May-workshop',
        options = {'build_exe': build_exe_options},
        executables = [exe])