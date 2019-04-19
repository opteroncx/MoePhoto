import os
from python import moe_utils

# recompile files
os.system('npm run build')
moe_utils.compile_pyc()
os.system('python setup.py build')
print('### fix scipy ####')
os.rename('./build/exe.win-amd64-3.6/lib/scipy/spatial/cKDTree.cp36-win_amd64.pyd','./build/exe.win-amd64-3.6/lib/scipy/spatial/ckdtree.cp36-win_amd64.pyd')