# -*- coding:utf-8 -*-
import argparse
import torch
import os
import cv2
from PIL import Image
from torch.autograd import Variable
import numpy as np
from functools import partial
import pickle
import readgpu
import psutil
import traceback
from turbo import Net2x, Net3x, Net4x
# from collections import OrderedDict
from config import Config

parser = argparse.ArgumentParser(description="MoePhoto")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/sr24/model.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--net", default='null', type=str, help="network file")

cuda = torch.cuda.is_available()

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def sr(im, opt):
    print("doing super resolution")
    img_size = opt.cropsize
    x = check_rgba(im)
    sc = opt.scale
    # tmp = resize_image_by_pil(x, sc)
    # gt_yuv = convert_rgb_to_ycbcr(tmp)
    # x = convert_rgb_to_y(x)
    x = torch.tensor(x, dtype=torch.float32)

    # erroneous if image size < cropsize, but no idea about what to pad
    if not (x.shape[0] == img_size and x.shape[1] == img_size):
        num_across = x.shape[0] // img_size
        if x.shape[0] % img_size != 0:
            num_across += 1
        num_up = x.shape[1] // img_size
        if x.shape[1] % img_size != 0:
            num_up += 1
        tmp_image = torch.zeros([x.shape[0] * sc, x.shape[1] * sc, 3])
        leftS = x.shape[0]
        leftT = leftS * sc
        for i in range(num_across):
            leftS -= img_size
            if leftS < 0:
                leftS = 0
            leftT = leftS * sc
            topS = x.shape[1]
            for j in range(num_up):
                topS -= img_size
                if topS < 0:
                    topS = 0
                topT = topS * sc
                s = x[leftS:leftS + img_size, topS:topS + img_size]
                # tmp = predict(s, opt)
                tmp = torch.stack([predict(s[:,:,0], opt), predict(s[:,:,1], opt), predict(s[:,:,2], opt)], dim=2)
                tmp_image[leftT:leftT + tmp.shape[0], topT:topT + tmp.shape[1]] = tmp
    else:
        tmp_image = predict(x, opt)

    recon = tmp_image.numpy() # convert_y_and_cbcr_to_rgb(tmp_image.numpy(), gt_yuv[:, :, 1:3])

    return recon

def getModel(opt):

    if opt.modelCached != None:
      return opt.modelCached
    if opt.scale == 2:
        print('loading net 2x')
        model = Net2x()
    elif opt.scale == 3:
        print('loading net 3x')
        model = Net3x()
    elif opt.scale == 4:
        print('loading net 4x')
        model = Net4x()

    pickle.load = partial(pickle.load, encoding="utf-8")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="utf-8")
    weights = torch.load(opt.model, map_location=lambda storage, loc: storage, pickle_module=pickle)
    # weights = torch.load(opt.model)
    print('reloading weights')
    model.load_state_dict(weights)

    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    return model

def predict(img_read, opt):
    im_input = img_read / 255.
    im_input = Variable(im_input).view(1, -1, im_input.shape[0], im_input.shape[1])
    if cuda:
        im_input = im_input.cuda()

    model = getModel(opt)

    HR = model(im_input)
    HR = HR[-1]
    im_h_y = HR.data[0].to(dtype=torch.float32)

    im_h_y = im_h_y * 256.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    if len(im_h_y.shape) == 3:
        im_h_y = im_h_y[0, :, :]

    return im_h_y.cpu()

def convert_rgb_to_y(image, jpeg_mode=False, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image

def convert_rgb_to_ycbcr(image, jpeg_mode=False, max_value=255):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, [1, 2]] += max_value / 2
    else:
        xform = np.array(
            [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
             [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
        ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=False, max_value=255.0):
    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image)

def convert_ycbcr_to_rgb(ycbcr_image):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
    rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
    xform = np.array(
        [[298.082 / 256.0, 0, 408.583 / 256.0],
         [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
         [298.082 / 256.0, 516.412 / 256.0, 0]])
    rgb_image = rgb_image.dot(xform.T)
    return rgb_image

def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA images
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image

def check_rgba(im):
    '''convert rgba2rgb'''
    if im.shape[2]==4:
        rgb = im[:,:,0:3]
    else:
        rgb = im
    return rgb

##################################
def main():
    print('请在flask内运行')

def getOpt(scale, mode):
    opt = parser.parse_args()
    mode_switch = {
        'a2': './model/a2/model_new.pth',
        'a3': './model/a3/model_new.pth',
        'a4': './model/a4/model_new.pth',
        'p2': './model/p2/model_new.pth',
        'p3': './model/p3/model_new.pth',
        'p4': './model/p4/model_new.pth',
    }
    nmode = mode+str(scale)
    if not (nmode in mode_switch):
        return {}
    opt.model = mode_switch[nmode]
    opt.scale = scale
    if cuda:
        torch.cuda.empty_cache()

    conf = Config().getConfig()
    if conf[0] == 0:
        if cuda:
            free_ram = readgpu.getGPU()
            if scale == 2:
                cropsize = int(np.sqrt((free_ram)/0.015))
            elif scale ==3:
                cropsize = int(np.sqrt((free_ram)/0.06))
            elif scale ==4:
                cropsize = int(np.sqrt((free_ram)/0.12))
        else:
            mem = psutil.virtual_memory()
            free_ram = mem.free
            free_ram = int(free_ram/1024**2)
            # 预留内存防止系统卡死
            free_ram -= 300
            # torch.set_num_threads(1)
            if scale == 2:
                cropsize = int(np.sqrt((free_ram)/0.05))
            elif scale ==3:
                cropsize = int(np.sqrt((free_ram)/0.12))
            elif scale ==4:
                cropsize = int(np.sqrt((free_ram)/0.24))
    else:
        cropsize = conf[0]

    if cropsize > 1024:
        cropsize = 1024
    opt.cropsize = cropsize
    print('当前SR切块大小：',cropsize)
    opt.modelCached = None
    opt.modelCached = getModel(opt)
    return opt

def dosr(im, scale, mode):
    opt = getOpt(scale, mode)
    sim = None

    try:
        sim = sr(im, opt)
    except Exception as msg:
        print('错误内容=='+str(msg))
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
    return sim

if __name__=="__main__":
    main()