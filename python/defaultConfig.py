# MoePhoto运行配置文件
defaultConfig = {
  'crop_sr': ('auto', '放大模式\n示例-分块大小320像素：\ncrop_sr: 320'),
  'crop_dn': ('auto', '普通降噪'),
  'crop_dns': ('auto', '风格化、强力降噪'),
  'videoName': ('out_{timestamp}.mkv', '输出视频文件名'),
  'maxMemoryUsage': (0, '最大使用的内存MB'),
  'maxGraphicMemoryUsage': (0, '最大使用的显存MB'),
  'cuda': (True, '使用CUDA'),
  'fp16': (True, '使用半精度浮点数'),
  'deviceId': (0, '使用的GPU序号'),
  'defaultDecodec': ('', '默认视频输入解码选项'),
  'defaultEncodec': ('libx264 -pix_fmt yuv420p', '默认视频输出编码选项'),
  'ensembleSR': (0, '放大时自集成的扩增倍数， 0-7'),
  'outDir': ('download', '输出目录'),
  'uploadDir': ('upload', '上传目录'),
  'sharedMemSize': (100 * 2 ** 20, '前后台共享的内存文件交换区字节大小，要能装下png格式的一张输入或输出图片'),
  'port': (2333, '监听端口')
}