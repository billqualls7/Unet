# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-11-17 10:46:56
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-11-17 10:47:18
# FilePath: \UnetV3\src\onnxopt.py
# Description: onnx 优化
# '''



import os
import torch
import torchvision
import onnx
from onnxsim import simplify
import onnxoptimizer

import tools
import sys
sys.path.append("./unets")
from nets import read_yaml

onnxpath = 'run\onnx\exp0\model_pruned.onnx'

unsimmode = 'model.onnx'
# simmode  = 
model = onnx.load(onnxpath)

passes = ["fuse_bn_into_conv"]
model = onnxoptimizer.optimize(model, passes)


# model, check = simplify(model)
onnx.save(model, 'run\onnx\exp0\model_pruned3.onnx')