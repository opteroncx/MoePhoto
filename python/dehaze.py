from torchvision.transforms import Normalize
import numpy as np
from models import AODnet
from sun_demoire import Net as SUNNet
from moire_obj import Net as ObjNet
from moire_screen_gan import Net as GANNet
from MPRNet import MPRNet
from NAFNet import NAFNet
from AiLUT import AiLUT
from imageProcess import initModel, Option
from config import config
normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
ramCoef = .95 / np.array([[1329., 480., 456.], [1509.3, 826.02, 828.], [69981, 9616, 5040], [30069, 3960, 2120],
                          [2620., 696., 457.], [5236., 1165., 692.], [15, 44, 44]])
ram2 = np.array([[[18196, 32868., 25/32], [-89<<20, 6336., 533 / 768], [-68<<22, 7264., 282/829]],
[[-98<<16, 6640., 1/771], [68<<19, 1152., 0], [53<<18, 1088., 0]]])
mode_switch = {
  'dehaze': ('./model/dehaze/AOD_net_epoch_relu_10.pth', AODnet, ramCoef[0], 1, 8),
  'sun': ('./model/demoire/sun_epoch_200.pth', SUNNet, ramCoef[1], 9, 32),
  'moire_obj': ('./model/demoire/moire_obj.pth', ObjNet, ram2[0], 9, 128),
  'moire_screen_gan': ('./model/demoire/moire_screen_gan.pth', GANNet, ram2[1], 17, 512),
  'MPRNet_deblurring': ('./model/MPRNet/model_deblurring.pth', MPRNet, ramCoef[2], 7, 8),
  'MPRNet_deraining': ('./model/MPRNet/model_deraining.pth', (lambda: MPRNet(n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16)), ramCoef[3], 7, 8),
  'NAFNet_deblur_32': ('./model/NAFNet/NAFNet-GoPro-width32.pth', (lambda: NAFNet(width=32, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])), ramCoef[4], 15, 16),
  'NAFNet_deblur_64': ('./model/NAFNet/NAFNet-GoPro-width64.pth', (lambda: NAFNet(width=64, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])), ramCoef[5], 15, 16),
  'NAFNet_deblur_JPEG_64': ('./model/NAFNet/NAFNet-REDS-width64.pth', (lambda: NAFNet(width=64, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])), ramCoef[5], 15, 16),
  'AiLUT_sRGB_3': ('./model/AiLUT/AiLUT-FiveK-sRGB.pth', AiLUT, ramCoef[6], 1, 8),
  'AiLUT_XYZ_3': ('./model/AiLUT/AiLUT-FiveK-XYZ.pth', AiLUT, ramCoef[6], 1, 8),
  'AiLUT_sRGB_5': ('./model/AiLUT/AiLUT-PPR10KA-sRGB.pth', (lambda: AiLUT(n_ranks=5, backbone='res18')), ramCoef[6], 1, 8),
}

def getOpt(optDe):
  model = optDe.get('model', 'dehaze')
  opt = Option()
  modelPath, opt.modelDef, ramCoef, opt.padding, opt.align = mode_switch[model]
  if model == 'dehaze':
    opt.prepare = normalize
  opt.strength = optDe.get('strength', 1.0)
  opt.ramCoef = ramCoef[config.getRunType()]
  opt.model = modelPath
  opt.modelCached = initModel(opt, modelPath, model)
  return opt