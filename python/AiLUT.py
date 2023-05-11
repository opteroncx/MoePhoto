"""
https://github.com/ImCharlesY/AdaInt
@InProceedings{yang2022adaint,
  title={AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement},
  author={Yang, Canqian and Jin, Meiguang and Jia, Xu and Xu, Yi and Chen, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock as ResBlock

from ailut import ailut_transform

class BasicBlock(nn.Sequential):
  r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
  """
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
    body = [
      nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
      nn.LeakyReLU(0.2)
    ]
    if norm:
      body.append(nn.InstanceNorm2d(out_channels, affine=True))
    super(BasicBlock, self).__init__(*body)

class TPAMIBackbone(nn.Sequential):
  r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

  Args:
    extra_pooling (bool, optional): Whether to insert an extra pooling layer
      at the very end of the module to reduce the number of parameters of
      the subsequent module. Default: False.
  """
  def __init__(self, extra_pooling=False):
    body = [
      BasicBlock(3, 16, stride=2, norm=True),
      BasicBlock(16, 32, stride=2, norm=True),
      BasicBlock(32, 64, stride=2, norm=True),
      BasicBlock(64, 128, stride=2, norm=True),
      BasicBlock(128, 128, stride=2),
      nn.Dropout(p=0.5)
    ]
    if extra_pooling:
      body.append(nn.AdaptiveAvgPool2d(2))
    super().__init__(*body)
    self.out_channels = 128 * (4 if extra_pooling else 64)

class Res18Backbone(ResNet):
  r"""The ResNet-18 backbone.
  """
  def __init__(self, **_):
    super().__init__(ResBlock, [2, 2, 2, 2])
    self.fc = nn.Identity()
    self.out_channels = 512

class LUTGenerator(nn.Module):
  r"""The LUT generator module (mapping h).

  Args:
    n_colors (int): Number of input color channels.
    n_vertices (int): Number of sampling points along each lattice dimension.
    n_feats (int): Dimension of the input image representation vector.
    n_ranks (int): Number of ranks in the mapping h (or the number of basis LUTs).
  """
  def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
    super().__init__()

    # h0
    self.weights_generator = nn.Linear(n_feats, n_ranks)
    # h1
    self.basis_luts_bank = nn.Linear(
      n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

    self.dims = (n_vertices,) * n_colors

  def forward(self, x):
    weights = self.weights_generator(x)
    luts = self.basis_luts_bank(weights)
    return luts.view(x.shape[0], -1, *self.dims)

class AdaInt(nn.Module):
  r"""The Adaptive Interval Learning (AdaInt) module (mapping g).

  It consists of a single fully-connected layer and some post-process operations.

  Args:
    n_colors (int): Number of input color channels.
    n_vertices (int): Number of sampling points along each lattice dimension.
    n_feats (int): Dimension of the input image representation vector.
    adaint_share (bool, optional): Whether to enable Share-AdaInt. Default: False.
  """

  def __init__(self, n_colors, n_vertices, n_feats, adaint_share=False) -> None:
    super().__init__()
    repeat_factor = 1 if adaint_share else n_colors
    self.intervals_generator = nn.Linear(n_feats, (n_vertices - 1) * repeat_factor)

    self.n_colors = n_colors
    self.n_vertices = n_vertices
    self.adaint_share = adaint_share

  def forward(self, x):
    r"""Forward function for AdaInt module.

    Args:
      x (tensor): Input image representation, shape (b, f).
    Returns:
      Tensor: Sampling coordinates along each lattice dimension, shape (b, c, d).
    """
    x = x.view(x.shape[0], -1)
    intervals = self.intervals_generator(x).view(x.shape[0], -1, self.n_vertices - 1)
    if self.adaint_share:
      intervals = intervals.repeat_interleave(self.n_colors, dim=1)
    intervals = intervals.softmax(-1)
    vertices = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
    return vertices

Backbones = dict(
  tpami=lambda **kwargs: ((256, 256), TPAMIBackbone(**kwargs)),
  res18=lambda **kwargs: ((224, 224), Res18Backbone(**kwargs))
)

class AiLUT(nn.Module):
  r"""Adaptive-Interval 3D Lookup Table for real-time image enhancement.

  Args:
    n_ranks (int, optional): Number of ranks in the mapping h
      (or the number of basis LUTs). Default: 3.
    n_vertices (int, optional): Number of sampling points along
      each lattice dimension. Default: 33.
    en_adaint (bool, optional): Whether to enable AdaInt. Default: True.
    en_adaint_share (bool, optional): Whether to enable Share-AdaInt.
      Only used when `en_adaint` is True. Default: False.
    backbone (str, optional): Backbone architecture to use. Can be either 'tpami'
      or 'res18'. Default: 'tpami'.
  """
  def __init__(self,
    n_ranks=3,
    n_vertices=33,
    en_adaint=True,
    en_adaint_share=False,
    backbone='tpami'):

    super().__init__()
    backbone_name = backbone.lower()
    # mapping f
    self.input_size, self.backbone = Backbones[backbone_name](extra_pooling=en_adaint)

    # mapping h
    self.lut_generator = LUTGenerator(3, n_vertices, self.backbone.out_channels, n_ranks)

    # mapping g
    self.adaint = AdaInt(3, n_vertices, self.backbone.out_channels, en_adaint_share)

  castDtype = 'autocast'

  def forward(self, imgs):
    r"""
    Args:
      img (Tensor): Input image, shape (b, c, h, w).
    Returns:
      Tensor: Output image, shape (b, c, h, w).
    """
    codes = F.interpolate(imgs, size=self.input_size, mode='bilinear', align_corners=False)
    # E: (b, f)
    codes = self.backbone(codes).view(imgs.shape[0], -1)
    # T: (b, c, d, d, d)
    luts = self.lut_generator(codes)
    # \hat{P}: (b, c, d)
    vertices = self.adaint(codes)

    return ailut_transform(imgs, luts, vertices)