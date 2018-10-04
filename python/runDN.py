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
from torchvision import transforms
import readgpu
import psutil
import traceback
from models import NetDN,SEDN
from config import Config

parser = argparse.ArgumentParser(description="SEDN")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/l15/model.pth", type=str, help="model path")

opt = parser.parse_args()
cuda = torch.cuda.is_available()

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def dn(im, opt):
    print("doing denoise")
    convert = True
    img_size = opt.cropsize
    x = check_rgba(im)
    if (len(x.shape) == 3) and not (x.shape[0] == img_size and x.shape[1] == img_size):
        num_across = x.shape[0] // img_size
        num_down = x.shape[1] // img_size
        tmp_image = np.zeros([x.shape[0], x.shape[1], 3])
        for i in range(num_across):
            for j in range(num_down):
                s = x[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size]
                tmp = s
                for channel in range(s.shape[2]):
                    t = s[:,:,channel]
                    output_clean_image = predict(t, convert, opt)
                    sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
                    tmp[:,:,channel] = sub_tmp[:,:]
                tmp_image[i * tmp.shape[0]:(i + 1) * tmp.shape[0], j * tmp.shape[1]:(j + 1) * tmp.shape[1]] = tmp
        if x.shape[0] % img_size != 0:
            for j in range(num_down):
                s = x[-1 * img_size:, j * img_size:(j + 1) * img_size]
                tmp = s
                for channel in range(s.shape[2]):
                    t = s[:,:,channel]
                    output_clean_image = predict(t, convert, opt)
                    sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
                    tmp[:,:,channel] = sub_tmp[:,:]
                tmp_image[-1 * tmp.shape[0]:, j * tmp.shape[1]:(j + 1) * tmp.shape[1]] = tmp
        if x.shape[1] % img_size != 0:
            for j in range(num_across):
                s = x[j * img_size:(j + 1) * img_size, -1 * img_size:]
                tmp = s
                for channel in range(s.shape[2]):
                    t = s[:,:,channel]
                    output_clean_image = predict(t, convert, opt)
                    sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
                    tmp[:,:,channel] = sub_tmp[:,:]
                tmp_image[j * tmp.shape[0]:(j + 1) * tmp.shape[0], -1 * tmp.shape[1]:] = tmp
        s = x[-1 * img_size:, -1 * img_size:]
        tmp = s
        for channel in range(s.shape[2]):
            t = s[:,:,channel]
            output_clean_image = predict(t, convert, opt)
            sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
            tmp[:,:,channel] = sub_tmp[:,:]
        tmp_image[-1 * tmp.shape[0]:, -1 * tmp.shape[1]:,] = tmp
    else:
        nim = np.zeros([x.shape[0], x.shape[1], 3])
        for i in range(x.shape[2]):
            tmp = predict(x, convert, opt)
            tmp_image[:,:,i] = tmp
    return tmp_image

def getModel(opt):
    if opt.modelCached != None:
      return opt.modelCached
    # pickle.load = partial(pickle.load, encoding="utf-8")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="utf-8")
    # model = torch.load(opt.model, map_location=lambda storage, loc: storage, pickle_module=pickle)["model"]
    if opt.model[:15] == './model/dn_lite':
        model = NetDN()
    else:
        model = SEDN()

    print(opt.model[:15])
    modelName = opt.model
    weights = torch.load(opt.model)
    print('reloading weights')
    model.load_state_dict(weights)

    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    return model

def predict(img_read, convert, opt):
    if convert:
        img_y = img_read
        img_y = img_y.astype("float32")
    else:
        im_gt_y, img_y = img_read
        im_gt_y = im_gt_y.astype("float32")
    im_input = img_y / 255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    if cuda:
        im_input = im_input.cuda()

    model = getModel(opt)
    DN = model(im_input)
    DN = DN[-1].cpu()
    im_h_y = DN.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h_y = im_h_y[0, :, :]
    recon = im_h_y
    return recon

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
    # if len(y_image.shape) <= 2:
    #     y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]
    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]
    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]
    return convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=jpeg_mode, max_value=max_value)

def convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=False, max_value=255.0):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    if jpeg_mode:
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array([[1, 0, 1.402], [1, - 0.344, - 0.714], [1, 1.772, 0]])
        rgb_image = rgb_image.dot(xform.T)
    else:
        rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - (16.0 * max_value / 256.0)
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array(
            [[max_value / 219.0, 0, max_value * 0.701 / 112.0],
             [max_value / 219, - max_value * 0.886 * 0.114 / (112 * 0.587), - max_value * 0.701 * 0.299 / (112 * 0.587)],
             [max_value / 219.0, max_value * 0.886 / 112.0, 0]])
        rgb_image = rgb_image.dot(xform.T)

    return rgb_image

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

def getOpt(model):
    opt = parser.parse_args()
    model_dict = {
        '15' : './model/l15/model_new.pth',
        '25' : './model/l25/model_new.pth',
        '50' : './model/l50/model_new.pth',
        'lite5' : './model/dn_lite5/model_new.pth',
        'lite10' : './model/dn_lite10/model_new.pth',
        'lite15' : './model/dn_lite15/model_new.pth'
    }
    if not(model in model_dict):
        return {}
    opt.model = model_dict[model]
<<<<<<< HEAD

    if cuda:
        torch.cuda.empty_cache()

=======
>>>>>>> 81071bccc1c0cc4561338d45f70007ace6c57476
    conf = Config().getConfig()
    if conf[1] == 0 or conf[2] == 0:
        if cuda:
            free_ram = readgpu.getGPU()
            if model[:4] == 'lite':
                if conf[1] == 0:
                    cropsize = int(np.sqrt((free_ram)/0.024))
                else:
                    cropsize = cropsize
            else:
                if conf[2] == 0:
                    cropsize = int(np.sqrt((free_ram)/0.016))
                else:
                    cropsize = conf[2]
        else:
            mem = psutil.virtual_memory()
            free_ram = mem.free
            free_ram = int(free_ram/1024**2)
            # 预留内存防止系统卡死
            free_ram -= 300
            # torch.set_num_threads(1)
            if model[:4] == 'lite':
                if conf[1] == 0:
                    cropsize = int(np.sqrt((free_ram)/0.042))
                else:
                    cropsize = conf[1]
            else:
                if conf[2]==0:
                    cropsize = int(np.sqrt((free_ram)/0.22))
                else:
                    cropsize = conf[2]
    else:
            if model[:4] == 'lite':
                cropsize = conf[1]
            else:
                cropsize = conf[2]

    opt.cropsize = cropsize
    print('当前denoise切块大小：', cropsize)
    opt.modelCached = None
    opt.modelCached = getModel(opt)
    return opt

def dodn(im, model):
    opt = getOpt(model)
    try:
        dim=dn(im, opt)
    except Exception as msg:
        print('错误内容=='+str(msg))
        trackback.print_exec()
    finally:
        torch.cuda.empty_cache()
    return dim

if __name__=="__main__":
    main()
