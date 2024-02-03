'''
Author: Wuyao 1955416359@qq.com
Date: 2023-10-13 21:00:01
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-02-01 21:14:21
FilePath: /UnetV3/src/infertrt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''




import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
from tools import *
from tools import load_image_as_array
from tools import get_angle
from torch2trt import TRTModule
import os
from data import *




if __name__ == '__main__':
    previous_left_fit = [0, 0, 0]
    previous_right_fit = [0, 0, 0]
    image_path = '../demo/002370.png'
    # 'run\onnx\exp1\model.onnx'
    # model_path = '../params/exp2/model.onnx'
    model_path = '../pre_model/UNet_Fire_trt.pth'
    net = TRTModule()
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print('successfully')
    
        # input_size = (256, 256)
        size = (320, 240)

        for i in range(30):
            
            time1 = time.time()
            frame=cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            heigh=int(frame.shape[0])
            width=int(frame.shape[1])
            # frame=frame[int(heigh*2/3):heigh,0:width]
            frame = Image.fromarray(np.uint8(frame))   

            temp = max(frame.size)
            mask = Image.new('RGB', (temp, temp))
            NewWidth=frame.size[0]
            NewHeigh=frame.size[1]
            mask.paste(frame, (0, NewWidth-NewHeigh))

            img = mask.resize(size)


            img_data=transform(img).cuda()
            out=net(img_data)
            out=torch.argmax(out,dim=1)
            out=torch.squeeze(out,dim=0)
            out=out.unsqueeze(dim=0)
            out=(out).permute((1,2,0)).cpu().detach().numpy()
            out=out*255.0
            vtherror=get_angle(out)



            time2 = time.time() 

            
            print("vtherror:",vtherror)
            # # print(time2-time1)
            print("fps:",1/(time2-time1))
    else:
        print('no loading')
