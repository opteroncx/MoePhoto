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

import numpy as np
import ncnn


class SRNet:
    def __init__(self, num_threads=8, use_gpu=True):
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        # self.mean_vals = [104.0, 117.0, 123.0]
        # self.norm_vals = []

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param("./realesrgan-x4plus.param")
        self.net.load_model("./realesrgan-x4plus.bin")
        # print("load net ready")

    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img.shape[1],
            img.shape[0],
            img.shape[1],
            img.shape[0],
        )
        # mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        # mat_in = ncnn.Mat(img.transpose(2,0,1))
        # print(mat_in.shape)
        mat_np = np.array(mat_in)/255.0
        # print(mat_np)
        mat_in = ncnn.Mat(mat_np)
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input("data", mat_in)

        ret, mat_out = ex.extract("output")

        # printf("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)

        out = np.array(mat_out).transpose(1,2,0)
        print(out)
        return out