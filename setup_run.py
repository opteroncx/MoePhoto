import os

os.system('python setup.py build')
os.rename('./build/exe.win-amd64-3.6/lib/scipy/spatial/cKDTree.cp36-win_amd64.pyd','./build/exe.win-amd64-3.6/lib/scipy/spatial/ckdtree.cp36-win_amd64.pyd')