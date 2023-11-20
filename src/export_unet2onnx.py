# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-10-06 20:19:28
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-11-17 10:45:24
# FilePath: \UnetV3\src\export_unet2onnx.py
# Description: 将模型导出为onnx模型
# '''



import os
import torch
import torchvision
import torch.onnx as onnx
import tools
import sys
sys.path.append("./unets")
from nets import read_yaml


def pth2onnx(weights):
    save_dir = 'run\onnx' 
    onnx_path = "model.onnx"
    save_path,_ = tools.mkdirr(save_dir)
    save_path_out = os.path.join(save_path,onnx_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.load_state_dict(torch.load(weights).state_dict())
    # net.load(weights)
    net = torch.load(weights)

    net.eval()
    # 创建一个虚拟输入
    input_shape = torch.randn(1, 3, 256, 256)
    input_shape = input_shape.to(device)
    output_names = ["output"]  # 输出节点的名称，可以根据实际情况修改
    # 将PyTorch模型转换为ONNX格式
    torch.onnx.export(
        net,
        input_shape,
        save_path_out,
        do_constant_folding=False,
        verbose=True,
        input_names=["input"],  # 输入节点的名称，可以根据实际情况修改
        output_names=output_names,
        opset_version=11  # 可选的 ONNX 版本号，根据需要指定
    )
    print(f'\033[92mTrans model successfully at {save_path_out}\033[0m')

if __name__ == '__main__':
    yamlpath = './cofig/train.yaml'
    net = read_yaml(yamlpath)[0]
    weights='params/exp1/min_loss.pt'  
    pth2onnx(weights)
