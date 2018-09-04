import PIL.Image as Image
import scipy.misc
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

cuda = torch.cuda.is_available()

def load_model():
    model = './model/dehaze/AOD_net_epoch_relu_10.pth'
    #===== load model =====
    print('===> Loading dehaze model')
    net = torch.load(model)
    if cuda:
        net = net.cuda()
    return net

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)

def run_test():
    net = load_model()
    input_image='./download/canyon1.jpg'
    output_filename='./download/canyon1_dh.jpg'
    #===== Load input image =====
    img = Image.open(input_image).convert('RGB')
    imgIn = transform(img).unsqueeze_(0)

    #===== Test procedures =====
    varIn = Variable(imgIn)
    if cuda:
        varIn = varIn.cuda()

    prediction = net(varIn)
    prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))
    scipy.misc.toimage(prediction).save(output_filename)

def Dehaze(img):
    net = load_model()
    print(img.shape)
    imgi = Image.fromarray(img)
    imgIn = transform(imgi).unsqueeze_(0)
    #===== Test procedures =====
    varIn = Variable(imgIn)
    if cuda:
        varIn = varIn.cuda()

    prediction = net(varIn)
    dhim = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))
    dhim = scipy.misc.toimage(dhim)   
    return np.array(dhim)

if __name__ == '__main__':
    print('dehaze')
    run_test()
