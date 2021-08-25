# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:35:24 2020

@author: LWS

An example of convert Pytroch model to onnx.
You should import your model and provide input according your model.
"""
import torch
import MoeNet_lite2,moire_obj,moire_screen_gan
import os

def dp_to_single_model(saved_state):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in saved_state.items():
        namekey = k[7:]  #去掉'module.'
        new_state_dict[namekey] = v
    return new_state_dict    

def get_onnx(model, onnx_save_path, example_tensor):

    example_tensor = example_tensor.cuda()

    _ = torch.onnx.export(model,  # model being run
                                  example_tensor,  # model input (or a tuple for multiple inputs)
                                  onnx_save_path,
                                  verbose=False,  # store the trained parameter weights inside the model file
                                  training=False,
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output']
                                  )

if __name__ == '__main__':
    precision = 'FP32'
    pretrained_path = '../model/lite/model.pth'
    # pretrained_weights = torch.load(pretrained_path)["model"].state_dict()
    pretrained_weights = torch.load(pretrained_path)
    # pretrained_weights = dp_to_single_model(pretrained_weights)
    model = MoeNet_lite2.Net().cuda()
    model.load_state_dict(pretrained_weights)
    onnx_save_path = os.path.join('./onnx2ncnn',precision+'_'+pretrained_path[2:-4]+'.onnx')
    onnx_save_dir = onnx_save_path.replace('model.onnx','')
    if not os.path.exists(onnx_save_dir):
        os.makedirs(onnx_save_dir)
    example_tensor = torch.randn(4, 1, 32, 32, device='cuda')

    # 导出模型
    get_onnx(model, onnx_save_path, example_tensor)

