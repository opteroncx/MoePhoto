import PIL.Image as Image
import scipy.misc
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from models import AODnet

cuda = torch.cuda.is_available()
net = False

def load_model():
  global net
  if net:
    return net
  model = './model/dehaze/AOD_net_epoch_relu_10.pth'
  #===== load model =====
  print('===> Loading dehaze model')
  net = AODnet()
  net.load_state_dict(torch.load(model))
  net.eval()
  for param in net.parameters():
    param.requires_grad_(False)
  if cuda:
    net = net.cuda()
  return net

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def run_test():
  net = load_model()
  input_image = './download/canyon1.jpg'
  output_filename = './download/canyon1_dh.jpg'
  #===== Load input image =====
  img = Image.open(input_image).convert('RGB')
  imgIn = transform(img).unsqueeze_(0)

  #===== Test procedures =====
  varIn = Variable(imgIn)
  if cuda:
    varIn = varIn.cuda()

  prediction = net(varIn)
  prediction = prediction.data.cpu().numpy().squeeze().transpose((1, 2, 0))
  scipy.misc.toimage(prediction).save(output_filename)

def Dehaze(img):
  net = load_model()
  print(img.shape)
  #imgi = Image.fromarray(img)
  imgIn = transform(img).unsqueeze(0)
  #===== Test procedures =====
  if cuda:
    imgIn = imgIn.cuda()

  prediction = net(imgIn)
  dhim = prediction.squeeze().cpu()
  return dhim

if __name__ == '__main__':
  print('dehaze')
  run_test()
