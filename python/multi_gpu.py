'''
多GPU处理
Demo
'''
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
# from skimage import transform,color
from torch.utils.data import DataLoader


class DatasetFromImage(data.Dataset):
    def __init__(self, file_path, scale=2):
        super(DatasetFromImage, self).__init__()
        self.ims = os.listdir(file_path)
        self.file_path = file_path
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        tim = Image.open(os.path.join(file_path, self.ims[0]))
        h, w = tim.size
        self.trans_bic = transforms.Compose([
            transforms.Resize((h*scale, w*scale), Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        data = Image.open(os.path.join(self.file_path, self.ims[index]))
        bic_im = self.trans_bic(data)
        data = data.convert("YCbCr")
        data_y, cb, cr = data.split()
        data = self.trans(data_y)
        # batch must contain tensors, numbers, dicts or lists;
        # data_dict = {'bic':bic_im,'name':self.ims[index]}
        return data, bic_im, self.ims[index]

    def __len__(self):
        return len(self.ims)


def model_convert(path, scale, gpus=1):
    if gpus > 1:
        loadmultiGPU = True
        gids = [i for i in range(gpus)]
    else:
        loadmultiGPU = False

    if scale == 2:
        from models import Net2x as Net
    if scale == 3:
        from models import Net3x as Net
    elif scale == 4:
        from models import Net4x as Net
    model = Net()

    if loadmultiGPU and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=gids).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    # optionally resume from a checkpoint
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        weights = torch.load(path)
        # saved_state = weights.state_dict()
        model.load_state_dict(weights)
        # multi gpu loader之前存的模型好像去掉了这部分只有权重
        # if loadmultiGPU:
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in saved_state.items():
        #         namekey = 'module.'+k  # add `module.`
        #         new_state_dict[namekey] = v
        #         # load params
        #     model.load_state_dict(new_state_dict)
        # else:
        #     model.load_state_dict(saved_state)
    else:
        print("=> no checkpoint found at '{}'".format(path))
    return model


def multi_gpu_run(model, im_path, outpath, gpus):
    print('running with multi GPU')
    # 如果patch一样大，开这个会加速
    cudnn.benchmark = True
    # 输入2D Tensor=[Batch,Channel,H,W]
    dataset = DatasetFromImage(im_path)
    # 暂定一个GPU跑一张图
    loader = DataLoader(dataset=dataset, batch_size=gpus)
    trans = transforms.Compose([
        transforms.ToPILImage(),
    ])
    for iter, batch in enumerate(loader, 1):
        y = batch[0]
        bicubic = batch[1]
        names = batch[2]
        # print(y)
        if torch.cuda.is_available():
            y = y.cuda()
        else:
            y = y.cpu()
        im_h_y = model(y)
        print(im_h_y.shape)
        #得看下shape，此处应该有个循环，把batch分解成多个图再合成
        im_h_y = trans(im_h_y)
        bicubic = trans(bicubic).convert("YCbCr")
        y, cb, cr = bicubic.split()
        HR = Image.merge('YCbCr', (im_h_y, cb, cr))
        HR.save(outpath)


if __name__ == '__main__':
    print('running with multi GPU')
    gpus = 1
    inpath = './temp/out/'
    outpath = './temp/vsr/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    model = model_convert('./model/a2/model_new.pth', 2, gpus)
    print('load success')
    multi_gpu_run(model, inpath, outpath, gpus)
