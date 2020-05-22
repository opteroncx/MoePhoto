import py_compile 
import os
import shutil

def copyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile)) 

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
        py_compile.compile(path+item, doraise=True, optimize=2)

    pycs = os.listdir(path+'__pycache__')
    for pyc in pycs:
        true_name = pyc.split('.')[0]+'.pyc'
        print(true_name)
        os.rename(path+'__pycache__/'+pyc,path+'__pycache__/'+true_name)
        copyfile(path+'__pycache__/'+true_name,'./pyc/'+true_name)

def delete_files(path):
    if not os.path.exists(path):
        print('file not exist')
    else:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)


if __name__ == '__main__':
    compile_pyc()
