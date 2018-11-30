import torch
import torchvision.transforms as transforms
from models import AODnet

cuda = torch.cuda.is_available()
net = False
model = './model/dehaze/AOD_net_epoch_relu_10.pth'

def load_model():
  global net
  if net:
    return net
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

def Dehaze(img):
  net = load_model()
  print(img.shape)
  imgIn = transform(img).unsqueeze(0)
  if cuda:
    imgIn = imgIn.cuda()

  prediction = net(imgIn)
  dhim = prediction.squeeze().cpu()
  return dhim
