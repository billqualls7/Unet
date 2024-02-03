'''
Date: 2024-02-01 15:16:59
LastEditTime: 2024-02-03 22:33:02
FilePath: /tls_visualtask/Unet-3.4/src/export_unet2trt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import os
import torch
import torchvision
import torch.onnx as onnx
import tools
import sys
from tools import InitModel
import shutil
import torch
from torch2trt import torch2trt





# def pth2trt_volkdep():
    
def pth2trt(weights,net,save_path):
    try:
        trt_path_eng = net.model+'.engine'
        trt_path_pth = net.model+'.pth'
        save_path_out_eng = os.path.join(save_path,trt_path_eng)
        save_path_out_pth = os.path.join(save_path,trt_path_pth)
        
        unet = net.net
        unet.load_state_dict(torch.load(weights))
        unet = unet.eval().cuda()
        

        print(torch.__version__)

        input_shape = torch.randn(1, 3, 240, 320).cuda()
        # convert to TensorRT feeding sample data as input
        model_trt = torch2trt(unet, [input_shape], fp16_mode=True)
        y = unet(input_shape)
        y_trt = model_trt(input_shape)
        print(torch.max(torch.abs(y - y_trt)))
        
        torch.save(model_trt.state_dict(), save_path_out_pth)
        # save(model_trt, save_path_out)
        print('------------------------------------------------------------------')
        print(f'\033[92mTrans model successfully at {save_path_out_pth}\033[0m')
        print('------------------------------------------------------------------')

        with open(save_path_out_eng, "wb") as f:
            f.write(model_trt.engine.serialize())
        print('------------------------------------------------------------------')
        print(f'\033[92mTrans model successfully at {save_path_out_eng}\033[0m')
        print('------------------------------------------------------------------')

    except Exception as e:
        print(f"Failed to convert model: {e}")
        # shutil.rmtree(save_path)
        # shutil.rmtree(net.weight_path)



# def pth2trt(weights,net,save_path):
#     try:
#         trt_path = net.model
#         save_path_out = os.path.join(save_path,trt_path)
#         device = torch.device("cpu")
#         unet = net.net
#         unet.load_state_dict(torch.load(weights))
#         unet = unet.eval().cuda()
        

#         print(torch.__version__)

#         input_shape = torch.randn(1, 3, 240, 320).cuda()
#         # convert to TensorRT feeding sample data as input
#         model_trt = torch2trt(unet, [input_shape])
#         y = unet(input_shape)
#         y_trt = model_trt(input_shape)
#         print(torch.max(torch.abs(y - y_trt)))
#         torch.save(model_trt.state_dict(), save_path_out)
#         print('------------------------------------------------------------------')
#         print(f'\033[92mTrans model successfully at {save_path_out}\033[0m')
#         print('------------------------------------------------------------------')
#     except Exception as e:
#         print(f"Failed to convert model: {e}")
#         # shutil.rmtree(save_path)
#         # shutil.rmtree(net.weight_path)


if __name__ == '__main__':
    yamlpath = '../cofig/train.yaml'
    net = InitModel(yamlpath)
    weights='../pre_model/UNet_Fire_min_loss.pt'  # UNetV3_2_min_loss
    save_path = '../pre_model/'
    pth2trt(weights,net,save_path)
