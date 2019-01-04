import sys
import os.path
import glob
from PIL import Image
import numpy as np
import torch
sys.path.append('python')
import gan  # pylint: disable=E0401
sys.path.pop()

# pylint: disable=E1101
model_path = sys.argv[1]  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'images/*'

model = gan.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

for idx, path in enumerate(glob.glob(test_img_folder)):
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    # read image
    img = np.array(Image.open(path))
    img = img * 1.0 / 256
    img = torch.from_numpy(img).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = (output * 256.0)
    Image.fromarray(output).save('results/{:s}_rlt.png'.format(base))
