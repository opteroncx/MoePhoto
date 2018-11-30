import PIL.Image as Image
import scipy.misc
import sys
sys.path.append('./python')
from dehaze import load_model, transform, cuda  # pylint: disable=E0401

def run_test():
  net = load_model()
  input_image = './download/canyon1.jpg'
  output_filename = './download/canyon1_dh.jpg'
  #===== Load input image =====
  img = Image.open(input_image).convert('RGB')
  imgIn = transform(img).unsqueeze_(0)

  #===== Test procedures =====
  if cuda:
    imgIn = imgIn.cuda()

  prediction = net(imgIn)
  prediction = prediction.data.cpu().numpy().squeeze().transpose((1, 2, 0))
  scipy.misc.toimage(prediction).save(output_filename)

if __name__ == '__main__':
  print('dehaze')
  run_test()
