'''
多GPU处理
Demo
'''
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

def model_convert(path,scale,gpus = 1):
    if gpus> 1:
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
        model=nn.DataParallel(model,device_ids=gids).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    # optionally resume from a checkpoint
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        saved_state = checkpoint["model"].state_dict()
        # multi gpu loader
        if loadmultiGPU:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in saved_state.items():
                namekey = 'module.'+k # add `module.`
                new_state_dict[namekey] = v
                # load params
            model.load_state_dict(new_state_dict)
        else: 
            model.load_state_dict(saved_state)
    else:
        print("=> no checkpoint found at '{}'".format(path))
    return model

def multi_gpu_run(model,path):
    print('running with multi GPU')
    # 如果patch一样大，开这个会加速
    cudnn.benchmark = True

if __name__ == '__main__':
    print('running with multi GPU')
    gpus = 10
    gids = [i for i in range(gpus)]
    print(gids)