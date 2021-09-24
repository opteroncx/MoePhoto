import logging
import torch
from torch import nn as nn
from torch.nn import functional as F

from imageProcess import getPadBy32, doCrop, StreamState, identity
from models import DCNv2Pack, ResidualBlockNoBN, make_layer, conv2d311
from slomo import backWarp
from runSlomo import getOptS, setOutShape
from progress import Node

RefTime = 7
WindowSize = 1
NumFeat = 64
log = logging.getLogger('Moe')

conv2d713 = lambda in_channels, out_channels:\
  nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
conv2d110 = lambda in_channels, out_channels:\
  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

"""Basic Module for SpyNet."""
BasicModule = lambda: nn.Sequential(
  conv2d713(8, 32), nn.ReLU(inplace=False),
  conv2d713(32, 64), nn.ReLU(inplace=False),
  conv2d713(64, 32), nn.ReLU(inplace=False),
  conv2d713(32, 16), nn.ReLU(inplace=False),
  conv2d713(16, 2))

class SpyNet(nn.Module):
  """SpyNet architecture.
  Args:
    load_path (str): path for pretrained SpyNet. Default: None.
  """

  def __init__(self, *_, load_path=None):
    super(SpyNet, self).__init__()
    self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
    if load_path:
      self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    self.flow_warp = None
    self.size = None

  def preprocess(self, tensor_input):
    tensor_output = (tensor_input - self.mean) / self.std
    return tensor_output

  def forward(self, inp):
    inp = self.preprocess(inp)
    ref = [0] * 5 + [inp[:, 0]]
    supp = [0] * 5 + [inp[:, 1]]

    for i in range(len(ref) - 1, 0, -1):
      ref[i - 1] = F.avg_pool2d(input=ref[i], kernel_size=2, stride=2, count_include_pad=False)
      supp[i - 1] = F.avg_pool2d(input=supp[i], kernel_size=2, stride=2, count_include_pad=False)

    N, _, H, W = ref[0].shape
    flow = ref[0].new_zeros([N, 2, H >> 1, W >> 1])
    if not self.flow_warp or self.size != [H, W]:
      self.size = [H, W]
      self.flow_warp = []
      for r in ref:
        _, _, H, W = r.shape
        self.flow_warp.append(backWarp(W, H, device=flow.device, dtype=flow.dtype, padding_mode='border'))
      assert not (H & 31 or W & 31)

    for level in range(len(ref)):
      upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

      f = self.flow_warp[level]
      flow = self.basic_module[level](torch.cat([
        ref[level],
        f(supp[level], upsampled_flow),
        upsampled_flow
      ], 1)) + upsampled_flow

    return flow

class PCDAlignment(nn.Module):
  """Alignment module using Pyramid, Cascading and Deformable convolution
  (PCD). It is used in EDVR.
  Ref:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
  Args:
    num_feat (int): Channel number of middle features. Default: 64.
    deformable_groups (int): Deformable groups. Defaults: 8.
  """

  def __init__(self, num_feat=64, deformable_groups=8):
    super(PCDAlignment, self).__init__()

    # Pyramid has three levels:
    # L3: level 3, 1/4 spatial size
    # L2: level 2, 1/2 spatial size
    # L1: level 1, original spatial size
    self.offset_conv1 = nn.ModuleDict()
    self.offset_conv2 = nn.ModuleDict()
    self.offset_conv3 = nn.ModuleDict()
    self.dcn_pack = nn.ModuleDict()
    self.feat_conv = nn.ModuleDict()

    # Pyramids
    for i in range(3, 0, -1):
      level = f'l{i}'
      self.offset_conv1[level] = conv2d311(num_feat * 2, num_feat)
      if i == 3:
        self.offset_conv2[level] = conv2d311(num_feat, num_feat)
      else:
        self.feat_conv[level] = conv2d311(num_feat * 2, num_feat)
        self.offset_conv2[level] = conv2d311(num_feat * 2, num_feat)
        self.offset_conv3[level] = conv2d311(num_feat, num_feat)

      self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

    # Cascading dcn
    self.cas_offset_conv1 = conv2d311(num_feat * 2, num_feat)
    self.cas_offset_conv2 = conv2d311(num_feat, num_feat)
    self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

  def forward(self, nbr_feat_l, ref_feat_l):
    """Align neighboring frame features to the reference frame features.
    Args:
      nbr_feat_l (list[Tensor]): Neighboring feature list. It
        contains three pyramid levels (L1, L2, L3), each with shape (b, c, h, w).
      ref_feat_l (list[Tensor]): Reference feature list. It
        contains three pyramid levels (L1, L2, L3), each with shape (b, c, h, w).
    Returns:
      Tensor: Aligned features.
    """
    # Pyramids
    upsampled_offset, upsampled_feat = None, None
    for i in range(3, 0, -1):
      level = f'l{i}'
      offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
      offset = self.lrelu(self.offset_conv1[level](offset))
      if i == 3:
        offset = self.lrelu(self.offset_conv2[level](offset))
      else:
        offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
        offset = self.lrelu(self.offset_conv3[level](offset))

      feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
      if i < 3:
        feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
      if i > 1:
        feat = self.lrelu(feat)

      if i > 1:  # upsample offset and features
        # x2: when we upsample the offset, we should also enlarge the magnitude.
        upsampled_offset = self.upsample(offset) * 2
        upsampled_feat = self.upsample(feat)

    # Cascading
    offset = torch.cat([feat, ref_feat_l[0]], dim=1)
    offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
    feat = self.lrelu(self.cas_dcnpack(feat, offset))
    return feat

class TSAFusion(nn.Module):
  """Temporal Spatial Attention (TSA) fusion module.
  Temporal: Calculate the correlation between center frame and neighboring frames;
  Spatial: It has 3 pyramid levels, the attention is similar to SFT.
    (SFT: Recovering realistic texture in image super-resolution by deep
        spatial feature transform.)
  Args:
    num_feat (int): Channel number of middle features. Default: 64.
    num_frame (int): Number of frames. Default: 5.
    center_frame_idx (int): The index of center frame. Default: 2.
  """

  def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
    super(TSAFusion, self).__init__()
    self.center_frame_idx = center_frame_idx
    # temporal attention (before fusion conv)
    self.temporal_attn1 = conv2d311(num_feat, num_feat)
    self.temporal_attn2 = conv2d311(num_feat, num_feat)
    self.feat_fusion = conv2d110(num_frame * num_feat, num_feat)

    # spatial attention (after fusion conv)
    self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
    self.spatial_attn1 = conv2d110(num_frame * num_feat, num_feat)
    self.spatial_attn2 = conv2d110(num_feat * 2, num_feat)
    self.spatial_attn3 = conv2d311(num_feat, num_feat)
    self.spatial_attn4 = conv2d110(num_feat, num_feat)
    self.spatial_attn5 = conv2d311(num_feat, num_feat)
    self.spatial_attn_l1 = conv2d110(num_feat, num_feat)
    self.spatial_attn_l2 = conv2d311(num_feat * 2, num_feat)
    self.spatial_attn_l3 = conv2d311(num_feat, num_feat)
    self.spatial_attn_add1 = conv2d110(num_feat, num_feat)
    self.spatial_attn_add2 = conv2d110(num_feat, num_feat)

    self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

  def forward(self, aligned_feat):
    """
    Args:
      aligned_feat (Tensor): Aligned features with shape (b, n, c, h, w).
    Returns:
      Tensor: Features after TSA with the shape (b, c, h, w).
    """
    b, n, c, h, w = aligned_feat.size()
    # temporal attention
    embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
    embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
    embedding = embedding.view(b, n, -1, h, w)  # (b, n, c, h, w)

    corr_l = []  # correlation list
    for i in range(n):
      emb_neighbor = embedding[:, i, :, :, :]
      corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
      corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
    corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, n, h, w)
    corr_prob = corr_prob.unsqueeze(2).expand(b, n, c, h, w)
    corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, n*c, h, w)
    aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

    # fusion
    feat = self.lrelu(self.feat_fusion(aligned_feat))  # (b, c, h, w)

    # spatial attention
    attn = self.lrelu(self.spatial_attn1(aligned_feat))  # (b, c, h, w)
    attn_max = self.max_pool(attn)
    attn_avg = self.avg_pool(attn)
    attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))  # (b, c, h / 2, w / 2)
    # pyramid levels
    attn_level = self.lrelu(self.spatial_attn_l1(attn))
    attn_max = self.max_pool(attn_level)  # (b, c, h / 4, w / 4)
    attn_avg = self.avg_pool(attn_level)
    attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
    attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
    attn_level = self.upsample(attn_level)  # (b, c, h / 2, w / 2)

    attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
    attn = self.lrelu(self.spatial_attn4(attn))
    attn = self.upsample(attn)  # (b, c, h, w)
    attn = self.spatial_attn5(attn)
    attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
    attn = torch.sigmoid(attn)

    # after initialization, * 2 makes (attn * 2) to be close to 1.
    feat = feat * attn * 2 + attn_add
    return feat

ConvResidualBlocks = lambda num_in_ch=3, num_out_ch=64, num_block=15: nn.Sequential(
      conv2d311(num_in_ch, num_out_ch), nn.LeakyReLU(negative_slope=0.1, inplace=True),
      make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

class IconVSR(nn.Module):
  # IconVSR, proposed also in the BasicVSR paper

  def __init__(self, num_feat=NumFeat, num_block=30, temporal_padding=3):
    super().__init__()

    self.num_feat = num_feat
    self.temporal_padding = temporal_padding
    self.keyframe_stride = temporal_padding * 2 + 1

    # keyframe_branch
    self.edvr = EDVRFeatureExtractor(self.keyframe_stride, num_feat)
    # alignment
    self.spynet = SpyNet()

    # propagation
    self.backward_fusion = conv2d311(2 * num_feat, num_feat)
    self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

    self.forward_fusion = conv2d311(2 * num_feat, num_feat)
    self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

    # reconstruction
    self.upsample = Upsample(num_feat=num_feat)

    self.flow_warp = None
    self.size = None

  def get_flow(self, x):
    b, n, c, h, w = x.size()

    x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
    flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

    return flows_forward, flows_backward

  def get_keyframe_feature(self, x, keyframe_idx):
    if self.temporal_padding == 2:
      x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
    elif self.temporal_padding == 3:
      x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
    x = torch.cat(x, dim=1)

    num_frames = 2 * self.temporal_padding + 1
    feats_keyframe = {}
    for i in keyframe_idx:
        feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
    return feats_keyframe

  # args: x: Tensor(b, n, 3, h ,w), h & w align by 32
  def forward(self, x):
    b, n, _, h, w = x.size()

    if not self.flow_warp or self.size != x.shape[-2:]:
      self.flow_warp = backWarp(w, h, device=x.device, dtype=x.dtype)
      self.size = x.shape[-2:]

    keyframe_idx = list(range(0, n, self.keyframe_stride))
    if keyframe_idx[-1] != n - 1:
      keyframe_idx.append(n - 1)  # last frame is a keyframe

    # compute flow and keyframe features
    flows_forward, flows_backward = self.get_flow(x)
    feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

    # backward branch
    out_l = []
    feat_prop = x.new_zeros(b, self.num_feat, h, w)
    for i in range(n - 1, -1, -1):
      x_i = x[:, i, :, :, :]
      if i < n - 1:
        flow = flows_backward[:, i, :, :, :]
        feat_prop = self.flow_warp(feat_prop, flow)
      if i in keyframe_idx:
        feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
        feat_prop = self.backward_fusion(feat_prop)
      feat_prop = torch.cat([x_i, feat_prop], dim=1)
      feat_prop = self.backward_trunk(feat_prop)
      out_l.insert(0, feat_prop)

    # forward branch
    feat_prop = torch.zeros_like(feat_prop)
    for i in range(0, n):
      x_i = x[:, i, :, :, :]
      if i > 0:
        flow = flows_forward[:, i - 1, :, :, :]
        feat_prop = self.flow_warp(feat_prop, flow)
      if i in keyframe_idx:
        feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
        feat_prop = self.forward_fusion(feat_prop)

      feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
      feat_prop = self.forward_trunk(feat_prop)

      # upsample
      out = self.upsample(feat_prop)
      base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
      out += base
      out_l[i] = out

    return torch.stack(out_l, dim=1)[..., :4 * h, :4 * w]

Upsample = lambda *_, num_feat=NumFeat: nn.Sequential(
  conv2d311(num_feat, num_feat * 4),
  nn.PixelShuffle(2),
  nn.LeakyReLU(negative_slope=0.1, inplace=True),
  conv2d311(num_feat, 64 * 4),
  nn.PixelShuffle(2),
  nn.LeakyReLU(negative_slope=0.1, inplace=True),
  conv2d311(64, 64),
  nn.LeakyReLU(negative_slope=0.1, inplace=True),
  conv2d311(64, 3))

class EDVRFeatureExtractor(nn.Module):

  def __init__(self, num_input_frame, num_feat):

    super(EDVRFeatureExtractor, self).__init__()

    self.center_frame_idx = num_input_frame // 2

    # extrat pyramid features
    self.conv_first = conv2d311(3, num_feat)
    self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=64)
    self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
    self.conv_l2_2 = conv2d311(num_feat, num_feat)
    self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
    self.conv_l3_2 = conv2d311(num_feat, num_feat)

    # pcd and tsa module
    self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
    self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

    # activation function
    self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

  def forward(self, x):
    b, n, c, h, w = x.size()

    # extract features for each frame
    # L1
    feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
    feat_l1 = self.feature_extraction(feat_l1)
    # L2
    feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
    feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
    # L3
    feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
    feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

    feat_l1 = feat_l1.view(b, n, -1, h, w)
    feat_l2 = feat_l2.view(b, n, -1, h >> 1, w >> 1)
    feat_l3 = feat_l3.view(b, n, -1, h >> 2, w >> 2)

    # PCD alignment
    ref_feat_l = [  # reference feature list
      feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
      feat_l3[:, self.center_frame_idx, :, :, :].clone()
    ]
    aligned_feat = []
    for i in range(n):
      nbr_feat_l = [  # neighboring feature list
        feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
      ]
      aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
    aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

    # TSA fusion
    return self.fusion(aligned_feat)

class KeyFrameState():
  def __init__(self, window):
    self.window = window
    self.count = 0

  def getSize(self, size=1 << 30):
    return size

  def pull(self, last=None, *_, **__):
    return not last

  def popBatch(self, size=1, last=None):
    res = torch.zeros((size,), dtype=torch.bool)
    for i in range(-self.count % self.window, size, self.window):
      res[i] = True
    if last:
      res[-1] = True
    self.count += size
    return res

def getKeyframeFeature(opt, keyframe, isKeyFrame, **_):
  return [(doCrop(opt.edvr, torch.stack(w).unsqueeze(0)) if b else None) for w, b in zip(keyframe, isKeyFrame)]

def calcFlowBackward(opt, flowInp, last):
  b, _, __, h, w = flowInp.size() # [[x_i, x_i+1], ...]

  flows = opt.spynet(flowInp).view(b, 1, 2, h, w)
  out = list(flows)
  if last:
    out.append(None)
  return out

def calcBackward(opt, inp, flowInp, keyframeFeature, last):
  n, _, h, w = inp.shape
  feat_prop = inp.new_zeros(1, NumFeat, h, w) # batch, channel, height, width
  out = []
  if last: # require at least 2 backward reference frames
    out = [0, 0] # pad 2 empties for the last window
  for i in range(n - 1, -1, -1):
    if i < n - 1 or not last:
      feat_prop = opt.flow_warp(feat_prop, flowInp[i])
    if keyframeFeature[i] != None:
      feat_prop = torch.cat([feat_prop, keyframeFeature[i]], dim=1)
      feat_prop = doCrop(opt.backward_fusion, feat_prop)
    feat_prop = torch.cat([inp[i], feat_prop], dim=1)
    feat_prop = doCrop(opt.backward_trunk, feat_prop)
    feat_prop = inp[i]
    out.insert(0, feat_prop)
  return out # only window[0] for window in out is used

def calcFlowForward(opt, state, flowInp, **_):
  out = []
  if state.first:
    out.append(None)
    flowInp = flowInp[1:]
    state.first = 0
  b, _, __, h, w = flowInp.size()
  x = torch.empty_like(flowInp) # [[x_i+1, x_i], ...]
  x[:, 0] = flowInp[:, 1]
  x[:, 1] = flowInp[:, 0]

  flows = opt.spynet(x).view(b, 1, 2, h, w)
  out.extend(list(flows))
  return out

def calcForward(opt, inp, flowInp, keyframeFeature, backward, **_):
  n, _, h, w = inp.shape
  feat_prop = inp.new_zeros(1, NumFeat, h, w) # batch, channel, height, width
  out = []
  for i in range(n):
    if flowInp[i] != None:
      feat_prop = opt.flow_warp(feat_prop, flowInp[i])
    if keyframeFeature[i] != None:
      feat_prop = torch.cat([feat_prop, keyframeFeature[i]], dim=1)
      feat_prop = doCrop(opt.forward_fusion, feat_prop)
    feat_prop = torch.cat([inp[i], backward[i][0], feat_prop], dim=1)
    feat_prop = doCrop(opt.forward_trunk, feat_prop)
    out.append(feat_prop)
  return out

def doUpsample(opt, inp, forward):
  setBatchSize(opt.upsample, inp)
  out = doCrop(opt.upsample, forward)
  base = F.interpolate(inp, scale_factor=4, mode='bilinear', align_corners=False)
  out += base
  return out

modelPath = './model/vsr/IconVSR_Vimeo90K_BDx4-cfcb7e00.pth'
# TODO: measure ram coefs
ramCoef = [.9 / x for x in (100., 100., 100., 100., 100., 1., 1., 100., 100., 100., 100., 100., 1., 1., 100., 100., 100., 100., 100.)]
fusionRamCoef = [.9 / x for x in (100., 100., 100.)]
newFusion = lambda *_: conv2d311(2 * NumFeat, NumFeat)
modules = dict(
  edvr={'weight': 'edvr', 'outShape': (1, NumFeat, 1, 1), 'staticDims': [0],
    'f': lambda *_: EDVRFeatureExtractor(RefTime, NumFeat)},
  spynet={'weight': 'spynet', 'f': SpyNet, 'streams': ['flowBackward', 'flowForward']},
  backward_trunk={'weight': 'backward_trunk', 'outShape': (1, NumFeat, 1, 1), 'staticDims': [0],
    'f': lambda *_: ConvResidualBlocks(NumFeat + 3, NumFeat, 30)},
  forward_trunk={'weight': 'forward_trunk', 'outShape': (1, NumFeat, 1, 1), 'staticDims': [0],
    'f': lambda *_: ConvResidualBlocks(2 * NumFeat + 3, NumFeat, 30)},
  upsample={'weight': 'upsample', 'outShape': (1, 3, 4, 4),
    'f': Upsample, 'streams': ['out']},
  backward_fusion={'weight': 'backward_fusion', 'outShape': (1, NumFeat, 1, 1), 'staticDims': [0],
    'f': newFusion, 'ramCoef': fusionRamCoef},
  forward_fusion={'weight': 'forward_fusion', 'outShape': (1, NumFeat, 1, 1), 'staticDims': [0],
    'f': newFusion, 'ramCoef': fusionRamCoef}
)

def getOpt(_):
  opt = getOptS(modelPath, modules, ramCoef)
  opt.flow_warp = None
  opt.i = 0
  return opt

def setBatchSize(opt, x):
  opt.outShape[0] = x.shape[0]

extend = lambda out, res: out.extend(tuple(res)) if res != None else None
def doVSR(func, node, opt):
  opt.inp = StreamState(offload=False)
  inp1 = StreamState()
  inp2 = StreamState()
  backwardInp = StreamState()
  flowInp = StreamState(2, offload=False)
  flowForwardInp = StreamState(offload=False).setPadding(1)
  flowBackwardInp = StreamState(offload=False)
  isKeyFrame = KeyFrameState(RefTime)
  opt.keyframeFeatureInp = StreamState(RefTime, tensor=False, reserve=1, offload=False)
  StreamState.pipe(identity, [opt.inp], [inp1, inp2, flowInp, backwardInp])
  StreamState.pipe(identity, [flowInp], [flowForwardInp, flowBackwardInp])
  keyframeFeature = StreamState(tensor=False, offload=False)
  n1 = Node({'IconVSR' :'KeyframeFeature'})
  StreamState.pipe(n1.bindFunc(getKeyframeFeature), [opt.keyframeFeatureInp, isKeyFrame], [keyframeFeature], args=[opt], size=7)
  keyframeFeature1 = StreamState(tensor=False)
  keyframeFeature2 = StreamState(tensor=False)
  StreamState.pipe(identity, [keyframeFeature], [keyframeFeature1, keyframeFeature2])
  flowBackward = StreamState(tensor=False)
  n2 = Node({'IconVSR' :'Flow'})
  opt.flowBackward = StreamState.pipe(n2.bindFunc(calcFlowBackward), [flowBackwardInp], [flowBackward], args=[opt], size=1)
  backward = StreamState(3, tensor=False)
  n3 = Node({'IconVSR' :'Backward'})
  StreamState.pipe(n3.bindFunc(calcBackward), [backwardInp, flowBackward, keyframeFeature1], [backward], args=[opt], size=20)
  flowForward = StreamState(tensor=False, offload=False)
  flowForward.first = 1 # signal alignment for frame 0, 1
  n4 = Node({'IconVSR' :'Flow'})
  opt.flowForward = StreamState.pipe(n4.bindFunc(calcFlowForward), [flowForwardInp], [flowForward], args=[opt, flowForward], size=1)
  forward = StreamState(offload=False)
  n5 = Node({'IconVSR' :'Forward'})
  StreamState.pipe(n5.bindFunc(calcForward), [inp1, flowForward, keyframeFeature2, backward], [forward], args=[opt])
  upsample = StreamState(store=False)
  n6 = Node({'IconVSR' :'upsample'})
  opt.out = StreamState.pipe(n6.bindFunc(doUpsample), [inp2, forward], [upsample], args=[opt], size=1)
  node.append(n1).append(n2).append(n3).append(n4).append(n5).append(n6)
  def f(x):
    node.reset()
    node.trace(0, p='VSR start')

    if opt.flow_warp is None:
      width, height, opt.pad, _ = getPadBy32(x, opt)
      opt.width = width << 2
      opt.height = height << 2
      opt.flow_warp = backWarp(width, height, device=x.device, dtype=x.dtype)
      opt.unpad = lambda im: im[:, :opt.width, :opt.height]
      setOutShape(modules, opt, height, width, lambda *_: 1)

    if opt.end:
      opt.keyframeFeatureInp.setPadding(opt.end)
      opt.end = 0
    if opt.start:
      opt.startPadding = opt.start
      opt.keyframeFeatureInp.setPadding(opt.start)
      opt.start = 0
    last = True if x is None else None
    if not last:
      x = opt.pad(x.unsqueeze(0))
      if opt.i + opt.startPadding >= RefTime >> 1:
        opt.inp.push(x)
      opt.keyframeFeatureInp.push(x)
      opt.i += 1
    out = []
    extend(out, opt.out.send(last))
    node.trace()
    while last:
      try:
        extend(out, opt.out.send(last))
      except StopIteration: break
    res = []
    for item in out:
      item = func(opt.unpad(item))
      if type(item) == list:
        res.extend(item)
      elif not item is None:
        res.append(item)
    return res
  return f