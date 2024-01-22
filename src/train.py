'''
Author: Wuyao 1955416359@qq.com
Date: 2023-11-03 19:19:26
LastEditors: Wuyao 1955416359@qq.com
LastEditTime: 2024-01-22 16:18:39
FilePath: /UnetV3/src/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''






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
import argparse
import numpy as np
from torchvision.transforms import ToPILImage

#----------------------------------------------------------
yamlpath = '../cofig/train.yaml'                  ##default 
#----------------------------------------------------------


train_loss_list = []
train_acc_list = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py", description='Unet train.py')
    # 给这个解析对象添加命令行参数
    parser.add_argument('--yamlpath', type=str, metavar='', default=yamlpath ,help='train params default:cofig/train.yaml')
    parser.add_argument('--onnx', type=bool, metavar='', default=False, help='creat onnx ? True or False default=False')
    parser.add_argument('--val', type=bool, metavar='', default=True, help='use val ? True or False default=True')
    args = parser.parse_args()  # 获取所有参数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _init = tools.InitModel(args.yamlpath)
    net = _init.net
    opt = _init.opt
    train_epch = _init.train_epch
    data_loader = _init.data_loader
    data_loader_val = _init.data_loader_val
    weight_path = _init.weight_path
    train_result_path = _init.train_result_path
    model_type = _init.model
    loss_fun = nn.CrossEntropyLoss()
    metric = tools.SegmentationMetric(6)
 

    # net,\
    # opt,\
    # train_epch,\
    # data_loader,\
    # weight_path,\
    # train_result_path,\
    # model_type = tools.InitModel(args.yamlpath)
    

    epoch = 0
    min_loss = float('inf') 
    best_val_loss = float('inf') 
    pbar = tqdm.tqdm(total=train_epch)
    starttime = time.time()
    while epoch < train_epch:
        
        net.train()
        for i, (image, segment_image) in enumerate((data_loader)):
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

            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{train_result_path}/{i}.png')


            
            print(f'{epoch}-{i}-train_loss===>>{train_loss.item():.10f}')
            # print(f'{epoch}-{i}-train_pa=====>>{pa:.12f}')
        # 验证模型
        if args.val and epoch > train_epch-5:
            net.eval()
            with torch.no_grad():
                # val_loss = 0.0
                for i, (val_image, segment_image) in enumerate(tqdm.tqdm(data_loader_val)):
                    val_image, segment_image = val_image.to(device), segment_image.to(device)
                    out = net(val_image)
                    out=torch.argmax(out,dim=1)
                    out=torch.squeeze(out,dim=0)
                    out=out.unsqueeze(dim=0)
                    out=(out).permute((1,2,0)).cpu().detach().numpy()
                    out=out*255.0
                    cv2.imwrite(f'{weight_path}/val/{i}.png',out)



        net.train()


                #     val_loss += loss_fun(val_outputs, segment_image.long())
                #     pred_labels = torch.argmax(val_outputs, dim=1).flatten().cpu()
                #     true_labels = segment_image.flatten().cpu()
                #     metric.addBatch(pred_labels, true_labels)
                #     pa = metric.pixelAccuracy()
                # val_loss /= len(data_loader_val)
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                # else:
                #     for param_group in opt.param_groups:
                #         param_group['lr'] *= 0.1  # 学习率衰减因子
                    
                

        TrainLossrecord = train_loss.item()
        train_loss_list.append(TrainLossrecord)
        # if args.val : train_acc_list.append(pa)
        if epoch > (train_epch//2):  #只保留后面的训练模型，前期训练模型损失值大，没有必要保存，增加了CPU和硬盘之间的IO操作，理论上会降低训练速度
            last_weight_path= os.path.join(weight_path, model_type+"_last.pt")
            torch.save(net, last_weight_path)
            if TrainLossrecord < min_loss:    #找出损失值最小的模型
                    min_loss = TrainLossrecord    
                    min_loss_weight_path = os.path.join(weight_path, model_type+"_min_loss.pt")
                    min_loss_round = epoch
                    torch.save(net.state_dict(), min_loss_weight_path)
            else: pass

            

        epoch += 1
        pbar.update(1)
    endtime = time.time()
    execution_time = round((endtime - starttime) / 60, 2)
    pbar.close()
    
    if args.onnx :pth2onnx(min_loss_weight_path, net, weight_path)   
    tools.train_print(min_loss_weight_path, min_loss, min_loss_round,execution_time)
    tools.draw("loss", train_epch,train_loss_list,weight_path)
    # if args.val: tools.draw("acc", train_epch,train_acc_list,weight_path)
    




