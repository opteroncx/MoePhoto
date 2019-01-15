import py_compile 
import os
import shutil


def compile_pyc(path = './python/'):
    if os.path.exists(path+'__pycache__'):
        shutil.rmtree(path+'__pycache__')

    items = os.listdir(path)
    try:
        items.remove('old_files')
    except:
        print('No deprecated old files')
    print(items)
    for item in items:    
        py_compile.compile(path+item)

    pycs = os.listdir(path+'__pycache__')
    for pyc in pycs:
        true_name = pyc.split('.')[0]+'.pyc'
        print(true_name)
        os.rename(path+'__pycache__/'+pyc,path+'__pycache__/'+true_name)

if __name__ == '__main__':
    compile_pyc()
