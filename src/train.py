# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-11-03 19:19:26
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-11-21 22:19:34
# FilePath: \UnetV3\src\train.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''



import os
import datetime
import tqdm
from torch import nn, optim
import torch
from torchvision.utils import save_image
import cv2
import subprocess
import time
import tools
from export_unet2onnx import pth2onnx
#----------------------------------------------------------
yamlpath = './cofig/train.yaml'                          ##
#----------------------------------------------------------


train_loss_list = []
ptname = []

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, opt, train_epch, data_loader, weight_path, train_result_path = tools.InitModel(yamlpath)
    loss_fun = nn.CrossEntropyLoss()
    epoch = 0
    min_loss = float('inf') 
    starttime = time.time()
    while epoch < train_epch:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            # cv2.imshow('image',image)
            # cv2.imshow('segment_image',segment_image)
            # print(segment_image.shape)
            out_image = net(image)
            # print(out_image.shape)
            train_loss = loss_fun(out_image, segment_image.long())
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{train_result_path}/{i}.png')
        
        if epoch % 1 == 0:
            TrainLossrecord=train_loss.item()
            train_loss_list.append(TrainLossrecord)

            if epoch > (train_epch/2):  #只保留后面的训练模型，前期训练模型损失值大，没有必要保存，增加了CPU和硬盘之间的IO操作，理论上会降低训练速度
                last_weight_path= os.path.join(weight_path,"last.pt")
                torch.save(net, last_weight_path)
                if TrainLossrecord < min_loss:    #找出损失值最小的模型
                     min_loss = TrainLossrecord    
                     min_loss_weight_path = os.path.join(weight_path, "min_loss.pt")
                     min_loss_round = epoch
                     torch.save(net, min_loss_weight_path)
                else: pass

            

        epoch += 1
    endtime = time.time()
    execution_time = round((endtime - starttime) / 60, 2)
    
    tools.save_to_excel(train_loss_list,weight_path)
    tools.train_print(min_loss_weight_path, min_loss, min_loss_round,execution_time)
    tools.draw(train_epch,train_loss_list,weight_path)
    pth2onnx(min_loss_weight_path)



