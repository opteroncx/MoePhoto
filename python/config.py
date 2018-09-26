# MoePhoto运行配置文件
class Config():
    def __init__(self):
        # 放大模式
        # 示例-分块大小320像素：
        # self.crop_sr = 320
        self.crop_sr = 'auto'
        # 普通降噪
        self.crop_dn = 'auto'
        # 风格化、强力降噪
        self.crop_dns = 'auto'

    def getConfig(self):
        if self.crop_sr == 'auto':
            sr = 0
        else:
            sr = self.crop_sr
        if self.crop_dn == 'auto':
            dn = 0
        else:
            dn = self.crop_dn
        if self.crop_dns == 'auto':
            dns = 0
        else:
            dns = self.crop_dns
        return sr,dn,dns
