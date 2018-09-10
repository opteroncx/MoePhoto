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
from models import NetDN,SEDN

parser = argparse.ArgumentParser(description="SEDN")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/l15/model.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

opt = parser.parse_args()
cuda = torch.cuda.is_available()

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def dn(im,cropsize=512):
    save = False
    eva = False
    convert = True
    img_size=cropsize
    x = im
    pimg = 'denoise'
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
                    output_clean_image = predict(t, save, convert, eva, pimg)
                    sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
                    tmp[:,:,channel] = sub_tmp[:,:]
                tmp_image[i * tmp.shape[0]:(i + 1) * tmp.shape[0], j * tmp.shape[1]:(j + 1) * tmp.shape[1]] = tmp
        if x.shape[0] % img_size != 0:
            for j in range(num_down):
                s = x[-1 * img_size:, j * img_size:(j + 1) * img_size]
                tmp = s
                for channel in range(s.shape[2]):
                    t = s[:,:,channel]
                    output_clean_image = predict(t, save, convert, eva, pimg)
                    sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
                    tmp[:,:,channel] = sub_tmp[:,:]
                tmp_image[-1 * tmp.shape[0]:, j * tmp.shape[1]:(j + 1) * tmp.shape[1]] = tmp
        if x.shape[1] % img_size != 0:
            for j in range(num_across):
                s = x[j * img_size:(j + 1) * img_size, -1 * img_size:]
                tmp = s
                for channel in range(s.shape[2]):
                    t = s[:,:,channel]
                    output_clean_image = predict(t, save, convert, eva, pimg)                        
                    sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
                    tmp[:,:,channel] = sub_tmp[:,:]
                tmp_image[j * tmp.shape[0]:(j + 1) * tmp.shape[0], -1 * tmp.shape[1]:] = tmp
        s = x[-1 * img_size:, -1 * img_size:]
        tmp = s
        for channel in range(s.shape[2]):
            t = s[:,:,channel]
            output_clean_image = predict(t, save, convert, eva, pimg)                
            sub_tmp = np.clip(output_clean_image, 0, 255).astype('uint8')
            tmp[:,:,channel] = sub_tmp[:,:]
        tmp_image[-1 * tmp.shape[0]:, -1 * tmp.shape[1]:,] = tmp
    else:
        nim = np.zeros([x.shape[0], x.shape[1], 3])
        for i in range(x.shape[2]):
            tmp = predict(x, save, convert, eva, pimg)
            tmp_image[:,:,i] = tmp
    return tmp_image

def predict(img_read, save, convert, eva, name):
    if convert:
        img_y = img_read
        img_y = img_y.astype("float32")
    else:
        im_gt_y, img_y = img_read
        im_gt_y = im_gt_y.astype("float32")
    im_input = img_y / 255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    # pickle.load = partial(pickle.load, encoding="utf-8")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="utf-8")
    # model = torch.load(opt.model, map_location=lambda storage, loc: storage, pickle_module=pickle)["model"]
    print(opt.model[:15])
    if opt.model[:15] == './model/dn_lite':
        model = NetDN()
    else:
        model = SEDN()
    
    weights = torch.load(opt.model)
    print('reloading weights')
    model.load_state_dict(weights)

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()   
    DN = model(im_input)
    DN = DN[-1].cpu()
    im_h_y = DN.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h_y = im_h_y[0, :, :]
    if save:
        recon = convert_y_and_cbcr_to_rgb(im_h_y, gt_yuv[:, :, 1:3])        
        save_figure(recon, name)
        if opt.mode == 'dn':
            print('save noise sample')
            noise = convert_y_and_cbcr_to_rgb(img_y,gt_yuv[:, :, 1:3])
            save_figure(noise,'noise_'+ name)
    print("doing denoise==>"+name)
    recon = im_h_y
    return recon

def save_figure(img,name):
    out_path='./out_dn/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print('saved '+name)
    cv2.imwrite(out_path+name[:-4]+'.png',img)

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

def dodn(im,model):
    model_dict = {
        '15' : './model/l15/model.pth',
        '25' : './model/l25/model.pth',
        '50' : './model/l50/model.pth',
        'lite5' : './model/dn_lite5/model_new.pth',
        'lite10' : './model/dn_lite10/model_new.pth',
        'lite15' : './model/dn_lite15/model_new.pth'
    }
    opt.model = model_dict[model]
    im = check_rgba(im)
    if cuda:
        free_ram = readgpu.getGPU()
        if model[:4] == 'lite':
            cropsize = int(np.sqrt((free_ram)/0.0115))
        else:
            cropsize = int(np.sqrt((free_ram)/0.075))        
    else:
        mem = psutil.virtual_memory()
        free_ram = mem.free 
        free_ram = int(free_ram/1024**2)
        # 预留内存防止系统卡死
        free_ram -= 300
        # torch.set_num_threads(1)
        if model[:4] == 'lite':
            cropsize = int(np.sqrt((free_ram)/0.042))
        else:
            cropsize = int(np.sqrt((free_ram)/0.22))

    print('cropsize==',cropsize)
    # try:
    dim=dn(im,cropsize)
    torch.cuda.empty_cache()
    # except Exception as msg:
    #     print('当前切块大小：',cropsize)
    #     print('出现错误，请重启程序，你的当前显存剩余',free_ram)
    #     print('错误内容=='+str(msg))
    #     torch.cuda.empty_cache()
    return dim

if __name__=="__main__":
    main()
