import torch
import torch.nn as nn
import torch.nn.functional as F

Conv2dS1 = lambda inC, outC, fSize: nn.Conv2d(inC, outC, fSize, stride=1, padding=(fSize - 1) // 2)
leaky_relu = nn.LeakyReLU(.1, inplace=True)

class down(nn.Module):
  """
  A class for creating neural network blocks containing layers:
  Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
  This is used in the UNet Class to create a UNet like NN architecture.
  ...
  Methods
  -------
  forward(x)
    Returns output tensor after passing input `x` to the neural network
    block.
  """
  def __init__(self, inChannels, outChannels, filterSize):
    """
    Parameters
    ----------
      inChannels : int
        number of input channels for the first convolutional layer.
      outChannels : int
        number of output channels for the first convolutional layer.
        This is also used as input and output channels for the
        second convolutional layer.
      filterSize : int
        filter size for the convolution filter. input N would create
        a N x N filter.
    """
    super(down, self).__init__()
    # Initialize convolutional layers.
    self.conv1 = Conv2dS1(inChannels, outChannels, filterSize)
    self.conv2 = Conv2dS1(outChannels, outChannels, filterSize)

  def forward(self, x):
    """
    Returns output tensor after passing input `x` to the neural network
    block.
    Parameters
    ----------
      x : tensor
        input to the NN block.
    Returns
    -------
      tensor
        output of the NN block.
    """
    # Average pooling with kernel size 2 (2 x 2).
    x = F.avg_pool2d(x, 2)
    # Convolution + Leaky ReLU
    x = leaky_relu(self.conv1(x))
    # Convolution + Leaky ReLU
    x = leaky_relu(self.conv2(x))
    return x

class up(nn.Module):
  """
  A class for creating neural network blocks containing layers:
  Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
  This is used in the UNet Class to create a UNet like NN architecture.
  ...
  Methods
  -------
  forward(x, skpCn)
    Returns output tensor after passing input `x` to the neural network
    block.
  """
  def __init__(self, inChannels, outChannels):
    """
    Parameters
    ----------
      inChannels : int
        number of input channels for the first convolutional layer.
      outChannels : int
        number of output channels for the first convolutional layer.
        This is also used for setting input and output channels for
        the second convolutional layer.
    """
    super(up, self).__init__()
    # Initialize convolutional layers.
    self.conv1 = Conv2dS1(inChannels, outChannels, 3)
    # (2 * outChannels) is used for accommodating skip connection.
    self.conv2 = Conv2dS1(outChannels * 2, outChannels, 3)

  def forward(self, x, skpCn):
    """
    Returns output tensor after passing input `x` to the neural network
    block.
    Parameters
    ----------
      x : tensor
        input to the NN block.
      skpCn : tensor
        skip connection input to the NN block.
    Returns
    -------
      tensor
        output of the NN block.
    """
    # Bilinear interpolation with scaling 2.
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    # Convolution + Leaky ReLU
    x = leaky_relu(self.conv1(x))
    # Convolution + Leaky ReLU on (`x`, `skpCn`)
    x = leaky_relu(self.conv2(torch.cat((x, skpCn), 1))) # pylint: disable=E1101
    return x

class UNet(nn.Module):
  """
  A class for creating UNet like architecture as specified by the
  Super SloMo paper.
  ...
  Methods
  -------
  forward(x)
    Returns output tensor after passing input `x` to the neural network
    block.
  """
  def __init__(self, inChannels, outChannels):
    """
    Parameters
    ----------
      inChannels : int
        number of input channels for the UNet.
      outChannels : int
        number of output channels for the UNet.
    """
    super(UNet, self).__init__()
    # Initialize neural network blocks.
    self.conv1 = Conv2dS1(inChannels, 32, 7)
    self.conv2 = Conv2dS1(32, 32, 7)
    self.down1 = down(32, 64, 5)
    self.down2 = down(64, 128, 3)
    self.down3 = down(128, 256, 3)
    self.down4 = down(256, 512, 3)
    self.down5 = down(512, 512, 3)
    self.up1   = up(512, 512)
    self.up2   = up(512, 256)
    self.up3   = up(256, 128)
    self.up4   = up(128, 64)
    self.up5   = up(64, 32)
    self.conv3 = Conv2dS1(32, outChannels, 3)

  def forward(self, x):
    """
    Returns output tensor after passing input `x` to the neural network.
    Parameters
    ----------
      x : tensor
        input to the UNet.
    Returns
    -------
      tensor
        output of the UNet.
    """
    x  = leaky_relu(self.conv1(x))
    s1 = leaky_relu(self.conv2(x))
    s2 = self.down1(s1)
    s3 = self.down2(s2)
    s4 = self.down3(s3)
    s5 = self.down4(s4)
    x  = self.down5(s5)
    x  = self.up1(x, s5)
    x  = self.up2(x, s4)
    x  = self.up3(x, s3)
    x  = self.up4(x, s2)
    x  = self.up5(x, s1)
    x  = leaky_relu(self.conv3(x))
    return x


class backWarp(nn.Module):
  """
  A class for creating a backwarping object.
  This is used for backwarping to an image:
  Given optical flow from frame I0 to I1 --> F_0_1 and frame I1,
  it generates I0 <-- backwarp(F_0_1, I1).
  ...
  Methods
  -------
  forward(x)
    Returns output tensor after passing input `img` and `flow` to the backwarping
    block.
  """
  def __init__(self, W, H, device, dtype=torch.float, padding_mode='zeros'): # pylint: disable=E1101
    """
    Parameters
    ----------
      W : int
        width of the image.
      H : int
        height of the image.
      device : device
        computation device (cpu/cuda).
    """
    super(backWarp, self).__init__()
    # create a grid
    gridY, gridX = torch.meshgrid(torch.arange(H), torch.arange(W)) # pylint: disable=E1101
    self.W = W
    self.H = H
    self.gridX = gridX.to(dtype=dtype, device=device) # pylint: disable=E1101
    self.gridY = gridY.to(dtype=dtype, device=device) # pylint: disable=E1101
    self.padding_mode = padding_mode

  def forward(self, img, flow):
    """
    Returns output tensor after passing input `img` and `flow` to the backwarping
    block.
    I0  = backwarp(I1, F_0_1)
    Parameters
    ----------
      img : tensor(n, c, h, w)
        frame I1.
      flow : tensor(n, 2, h, w)
        optical flow from I0 and I1: F_0_1.
    Returns
    -------
      tensor
        frame I0.
    """
    # Extract horizontal and vertical flows.
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    x = self.gridX.unsqueeze(0).expand_as(u) + u
    y = self.gridY.unsqueeze(0).expand_as(v) + v
    # range -1 to 1
    x = 2*(x/self.W - 0.5)
    y = 2*(y/self.H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3) # pylint: disable=E1101
    # Sample pixels using bilinear interpolation.
    # set both here and interpolate's align_corners to False will occur memory overflow
    imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True, padding_mode=self.padding_mode, mode='bilinear')
    return imgOut

"""unused
# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1.
t = np.linspace(0.125, 0.875, 7)

def getFlowCoeff(indices, device):
  ""
  Gets flow coefficients used for calculating intermediate optical
  flows from optical flows between I0 and I1: F_0_1 and F_1_0.
  F_t_0 = C00 x F_0_1 + C01 x F_1_0
  F_t_1 = C10 x F_0_1 + C11 x F_1_0

  where,
  C00 = -(1 - t) x t
  C01 = t x t
  C10 = (1 - t) x (1 - t)
  C11 = -t x (1 - t)
  Parameters
  ----------
    indices : tensor
      indices corresponding to the intermediate frame positions
      of all samples in the batch.
    device : device
      computation device (cpu/cuda).
  Returns
  -------
    tensor
      coefficients C00, C01, C10, C11.
  ""
  # Convert indices tensor to numpy array
  ind = indices.detach().numpy()
  C11 = C00 = - (1 - (t[ind])) * (t[ind])
  C01 = (t[ind]) * (t[ind])
  C10 = (1 - (t[ind])) * (1 - (t[ind]))
  return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)

def getWarpCoeff(indices, device):
  ""
  Gets coefficients used for calculating final intermediate
  frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.
  It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)
  where,
  C0 = 1 - t
  C1 = t

  V_t_0, V_t_1 --> visibility maps
  g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

  Parameters
  ----------
    indices : tensor
      indices corresponding to the intermediate frame positions
      of all samples in the batch.
    device : device
      computation device (cpu/cuda).
  Returns
  -------
    tensor
      coefficients C0 and C1.
  ""
  # Convert indices tensor to numpy array
  ind = indices.detach().numpy()
  C0 = 1 - t[ind]
  C1 = t[ind]
  return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device)
"""
