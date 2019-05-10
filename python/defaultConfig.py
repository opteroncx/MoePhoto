# MoePhoto运行配置文件
defaultConfig = {
  'version': ('4.7.0',), # distribution build doesn't have package.json
  'crop_sr': ('auto',),
  'crop_dn': ('auto',),
  'crop_dns': ('auto',),
  'videoName': ('out_{timestamp}.mkv',),
  'maxMemoryUsage': (0,),
  'maxGraphicMemoryUsage': (0,),
  'cuda': (True,),
  'fp16': (True,),
  'deviceId': (0,),
  'defaultDecodec': ('',),
  'defaultEncodec': ('libx264 -pix_fmt yuv420p',),
  'ensembleSR': (0,),
  'outDir': ('download',),
  'uploadDir': ('upload',),
  'logPath': ('.user/log.txt',),
  'opsPath': ('.user/ops.json',),
  'videoPreview': ('jpeg',),
  'maxResultsKept': (1 << 10,),
  'sharedMemSize': (100 * 2 ** 20, '前后台共享的内存文件交换区字节大小，要能装下一张输入或输出图片'),
  'port': (2333,)
}