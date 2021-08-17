# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys
import cv2
import numpy as np
import ncnn
from ncnn.model_zoo import get_model
from ncnn.utils import print_topk
from SRNet import SRNet
from PIL import Image
from skimage import io,color

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: %s [imagepath]\n" % (sys.argv[0]))
    #     sys.exit(0)

    # imagepath = sys.argv[1]
    imagepath = './input2.jpg'

    m = cv2.imread(imagepath)
    if m is None:
        print("cv2.imread %s failed\n" % (imagepath))
        sys.exit(0)

    # net = get_model("squeezenet", num_threads=4, use_gpu=True)
    net = SRNet()

    res = net(m)

    # print_topk(cls_scores, 5)
    # print(res)
    # res = cv2.cvtColor(res,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('sr.png',(res*255).astype(np.uint8))
    
    # BGR2RGB
    nim = np.zeros([1024,1024,3])
    nim[:,:,2] = res[:,:,0] 
    nim[:,:,1] = res[:,:,1]
    nim[:,:,0] = res[:,:,2]
    io.imsave('sr.png',nim)
