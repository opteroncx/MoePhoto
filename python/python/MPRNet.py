# Syed Waqas Zamir https://github.com/swz30/MPRNet
import torch
import torch.nn as nn

from models import Conv3x3, FRM
##########################################################################
conv = lambda in_channels, out_channels, kernel_size, bias=False, stride = 1:\
  nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size >> 1, bias=bias, stride = stride)

class Residual(nn.Sequential):
  def forward(self, x): return super().forward(x) + x
##########################################################################
## Channel Attention Block (CAB)
CAB = lambda n, k, r, bias, act:\
  Residual(conv(n, n, k, bias), act, conv(n, n, k, bias), FRM(n, r, bias))

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
  def __init__(self, n_feat, kernel_size, bias):
    super(SAM, self).__init__()
    self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
    self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
    self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

  def forward(self, x, x_img):
    x1 = self.conv1(x)
    img = self.conv2(x) + x_img
    x2 = torch.sigmoid(self.conv3(img))
    x1 = x1 * x2
    x1 = x1 + x
    return x1, img

##########################################################################
##---------- Resizing Modules ----------

UpSample = lambda in_channels, s_factor, sf=2: nn.Sequential(
  nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False),
  nn.Conv2d(in_channels + (s_factor if sf > 1 else 0), in_channels + (0 if sf > 1 else s_factor), 1, stride=1, padding=0, bias=False))
DownSample = lambda in_channels, s_factor: UpSample(in_channels, s_factor, 0.5)
class SkipUpSample(nn.Module):
  def __init__(self, in_channels, s_factor):
    super(SkipUpSample, self).__init__()
    self.up = UpSample(in_channels, s_factor)

  def forward(self, x, y):
    x = self.up(x)
    x = x + y
    return x

##########################################################################
## U-Net

class Encoder(nn.Module):
  def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
    super(Encoder, self).__init__()

    down = (
      nn.Identity(),
      DownSample(n_feat, scale_unetfeats),
      DownSample(n_feat + scale_unetfeats, scale_unetfeats)
    )

    self.encoder = nn.ModuleList(
      nn.Sequential(
        d,
        *(CAB(n_feat + scale_unetfeats * i, kernel_size, reduction, bias=bias, act=act) for _ in range(2))
      ) for i, d in enumerate(down)
    )

    # Cross Stage Feature Fusion (CSFF)
    if csff:
      self.csff_enc = nn.ModuleList((
        nn.Conv2d(n_feat,                       n_feat,                       kernel_size=1, bias=bias),
        nn.Conv2d(n_feat + scale_unetfeats,     n_feat + scale_unetfeats,     kernel_size=1, bias=bias),
        nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)
      ))
      self.csff_dec = nn.ModuleList((
        nn.Conv2d(n_feat,                       n_feat,                       kernel_size=1, bias=bias),
        nn.Conv2d(n_feat + scale_unetfeats,     n_feat + scale_unetfeats,     kernel_size=1, bias=bias),
        nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)
      ))
    else:
      self.csff_enc, self.csff_dec = None, None

  def forward(self, x, encoder_outs=None, decoder_outs=None):
    return encoderForward(self.encoder, self.csff_enc, self.csff_dec, x, encoder_outs, decoder_outs)

def encoderForward(encoder, csff_enc, csff_dec, x, encoder_outs, decoder_outs):
  enc = []
  for i in range(3):
    x = encoder[i](x)
    if encoder_outs is not None:
      x = x + csff_enc[i](encoder_outs[i]) + csff_dec[i](decoder_outs[i])
    enc.append(x)

  return enc

class Decoder(nn.Module):
  def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
    super(Decoder, self).__init__()

    self.decoder = nn.ModuleList(nn.Sequential(*(CAB(n_feat + scale_unetfeats * i, kernel_size, reduction, bias=bias, act=act) for _ in range(2))) for i in range(3))

    self.skip_attn = nn.ModuleList((
      CAB(n_feat,                   kernel_size, reduction, bias=bias, act=act),
      CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
    ))

    self.up = nn.ModuleList((SkipUpSample(n_feat, scale_unetfeats), SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)))

  def forward(self, outs):
    dec = [0 for _ in range(3)]
    for i in range(2, -1, -1):
      enc = outs[i]
      x = self.up[i](x, self.skip_attn[i](enc)) if i < 2 else enc
      x = self.decoder[i](x)
      dec[i] = x

    return dec

##########################################################################
## Original Resolution Block (ORB)
ORB = lambda n_feat, kernel_size, reduction, act, bias, num_cab:\
  Residual(*([CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)] + [conv(n_feat, n_feat, kernel_size)]))

##########################################################################
class ORSNet(nn.Module):
  def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
    super(ORSNet, self).__init__()

    self.orb = nn.ModuleList(ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab) for _ in range(3))

    genUps = lambda: (
      [],
      [UpSample(n_feat, scale_unetfeats)],
      [UpSample(n_feat + scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats)]
    )

    self.conv_enc = nn.ModuleList(nn.Sequential(*(up + [nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)])) for up in genUps())

    self.conv_dec = nn.ModuleList(nn.Sequential(*(up + [nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)])) for up in genUps())

  def forward(self, x, encoder_outs, decoder_outs):
    return encoderForward(self.orb, self.conv_enc, self.conv_dec, x, encoder_outs, decoder_outs)[2]

class MPRNet(nn.Module):
  def __init__(self, in_c=3, out_c=3, n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, reduction=4, bias=False):
    super(MPRNet, self).__init__()

    act = nn.PReLU()
    kernel_size = 3
    self.shallow_feat = nn.ModuleList(nn.Sequential(Conv3x3(in_c, n_feat, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act)) for _ in range(3))

    # Cross Stage Feature Fusion (CSFF)
    self.encoder = nn.ModuleList((
      Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False), # stage 1
      Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True), # stage 2
      ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab) # stage 3
    ))

    self.decoder = nn.ModuleList(Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats) for _ in range(2))

    self.sam = nn.ModuleList(SAM(n_feat, kernel_size=1, bias=bias) for _ in range(2))

    self.concat = nn.ModuleList(Conv3x3(n_feat * 2, feats, bias=bias) for feats in (n_feat, n_feat + scale_orsnetfeats))
    self.tail = Conv3x3(n_feat + scale_orsnetfeats, out_c, bias=bias)

  def stage(self, level, feat_0, feat_1, x_img):
    ##-------------------------------------------
    ##-------------- Stage level + 1---------------------
    ##-------------------------------------------
    ## Concat deep features
    feat = [torch.cat((p0, p1), 3 - level) for p0, p1 in zip(feat_0, feat_1)]

    ## Pass features through Decoder of Stage level + 1
    res = self.decoder[level](feat)

    ## Apply Supervised Attention Module (SAM)
    x_samfeats, stage_img = self.sam[level](res[0], x_img)

    ##-------------------------------------------
    ##-------------- Stage level + 2---------------------
    ##-------------------------------------------
    ## Compute Shallow Features
    x = self.shallow_feat[level + 1](x_img)

    ## Concatenate SAM features of Stage level + 1 with shallow features of Stage level + 2
    x_cat = self.concat[level](torch.cat([x, x_samfeats], 1))

    ## Process features of both patches with Encoder of Stage level + 2
    featE = self.encoder[level + 1](x_cat, feat, res)
    return featE, stage_img

  def forward(self, x3_img):
    # Original-resolution Image for Stage 3
    H, W = x3_img.shape[2:4]
    assert not (H & 7 or W & 7) # align to 8
    halfH, halfW = H >> 1, W >> 1

    # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

    # Two Patches for Stage 2
    x2top_img  = x3_img[:, :, :halfH]
    x2bot_img  = x3_img[:, :, halfH:]

    # Four Patches for Stage 1
    patches = x2top_img[:, :, :, :halfW], x2top_img[:, :, :, halfW:], x2bot_img[:, :, :, :halfW], x2bot_img[:, :, :, halfW:]

    ##-------------------------------------------
    ##-------------- Stage 1---------------------
    ##-------------------------------------------
    ## Compute Shallow Features
    ## Process features of all 4 patches with Encoder of Stage 1
    feat1_ltop, feat1_rtop, feat1_lbot, feat1_rbot = [self.encoder[0](self.shallow_feat[0](p)) for p in patches]

    feat2_top, stage1_img_top = self.stage(0, feat1_ltop, feat1_rtop, x2top_img)
    feat2_bot, stage1_img_bot = self.stage(0, feat1_lbot, feat1_rbot, x2bot_img)
    ## Output image at Stage 1
    # stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)

    x3_cat, stage2_img = self.stage(1, feat2_top, feat2_bot, x3_img)

    stage3_img = self.tail(x3_cat)

    return torch.clamp(stage3_img + x3_img, 0, 1)
    # return stage3_img + x3_img, stage2_img, stage1_img