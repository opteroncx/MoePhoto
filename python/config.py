# MoePhoto运行配置文件

config = {
    # 放大模式
    # 示例-分块大小320像素：
    # crop_sr: 320
    'crop_sr': 'auto',
    # 普通降噪
    'crop_dn': 'auto',
    # 风格化、强力降噪
    'crop_dns': 'auto',
    'frame_folder': './temp/video_frame',
    'srframe_folder': './temp/video_frame_sr',
    'video_out': './temp',
    'videoName': 'out.mkv'
}

class Config():
    def __init__(self): pass

    def getConfig(self):
        if config['crop_sr'] == 'auto':
            sr = 0
        else:
            sr = config['crop_sr']
        if config['crop_dn'] == 'auto':
            dn = 0
        else:
            dn = config['crop_dn']
        if config['crop_dns'] == 'auto':
            dns = 0
        else:
            dns = config['crop_dns']
        return sr, dn, dns

    def getPath(self):
        return config['frame_folder'], config['srframe_folder'], config['video_out'], config['videoName']
