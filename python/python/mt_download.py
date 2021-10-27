import requests
import threading
import datetime


def Handler(start, end, url, filename):

    headers = {'Range': 'bytes=%d-%d' % (start, end)}
    r = requests.get(url, headers=headers, stream=True)

    # 写入文件对应位置
    with open(filename, "r+b") as fp:
        fp.seek(start)
        var = fp.tell()
        fp.write(r.content)


def download_file(url, num_thread = 5,fname = ''):

    r = requests.head(url)
    try:
        if fname == '':
            file_name = url.split('/')[-1]
        else:
            file_name = fname
        file_size = int(r.headers['content-length'])   # Content-Length获得文件主体的大小，当http服务器使用Connection:keep-alive时，不支持Content-Length
    except:
        print("检查URL，或不支持多线程下载")
        return

    #  创建一个和要下载文件一样大小的文件
    fp = open(file_name, "wb")
    fp.truncate(file_size)
    fp.close()
    # 启动多线程写文件
    part = file_size // num_thread  # 如果不能整除，最后一块应该多几个字节
    for i in range(num_thread):
        start = part * i
        if i == num_thread - 1:   # 最后一块
            end = file_size
        else:
            end = start + part

        t = threading.Thread(target=Handler, kwargs={'start': start, 'end': end, 'url': url, 'filename': file_name})
        t.setDaemon(True)
        t.start()

    # 等待所有线程下载完成
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()
    print('%s 下载完成' % file_name)
if __name__ == '__main__':
    print('mt downloader')
    # download_file(url)