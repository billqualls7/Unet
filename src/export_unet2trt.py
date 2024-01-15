# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-11-25 20:26:27
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-11-25 20:26:47
# FilePath: \UnetV3\src\export_unet2trt.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''
import torch
from torch2trt import torch2trt

import os
import torch
import torchvision
import torch.onnx as onnx
import tools
import sys
sys.path.append("./unets")
from nets import read_yaml
import shutil
import onnx
import pycuda.driver as cuda
import pycuda.autoinit



def pth2trt(weights,unet):
    try:
        print(torch.__version__)
        save_dir = 'run/trt' 
        onnx_path = "model.engine"
        save_path,_ = tools.mkdirr(save_dir)
        save_path_out = os.path.join(save_path,onnx_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # net.load_state_dict(torch.load(weights).state_dict())
        # net.load(weights)
        # model  = torch.load(weights,map_location='cpu')
        # print(model)
        unet.load_state_dict(torch.load(weights))
        unet.eval().cuda()
        # builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        # network = builder.create_network()


        

        # 创建一个虚拟输入
        input_shape = torch.randn(1, 3, 240, 320).cuda()
        # input_shape = input_shape
        # output_names = ["output"]  # 输出节点的名称，可以根据实际情况修改
        
        model_trt = torch2trt(unet, [input_shape])
        torch.save(model_trt.state_dict(), save_path_out)
        # input_tensor = network.add_input(name="input", dtype=trt.float32, shape=input_shape)

        # trt_converter = trt.tensorrt.Converter(network, builder)
        # trt_converter.convert_pytorch(unet, "output")
        # network.mark_output(tensor=network.get_layer(network.num_layers - 1).get_output(0), name="output")
        # max_workspace_size = 1 << 30
        # engine = builder.build_cuda_engine(network)
        # with open(save_path_out, "wb") as f:
        #     f.write(engine.serialize())


        print(f'\033[92mTrans model successfully at {save_path_out}\033[0m')
    except Exception as e:
        print(f"Failed to convert model: {e}")
        shutil.rmtree(save_path)


if __name__ == '__main__':
    yamlpath = './cofig/train.yaml'
    net = read_yaml(yamlpath)[0]
    weights='params/exp13/min_loss.pt'  
    pth2trt(weights,net)