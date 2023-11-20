# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-07-22 14:52:24
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-07-22 14:54:41
# FilePath: \UnetV3\src\convolution_quantization.py
# Description: demo
# '''




import torch
import torch.quantization
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import torch
import time
from Unet import *
from utils import *
from data import *
from torchvision.utils import save_image

size = (256,256)
float_model = "params\demo\exp5\min_loss.pt"
image_path=r"F:\Code\UnetV2\trainDataset\JPEGImages\000030.png" 
frame=cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
heigh=int(frame.shape[0])
width=int(frame.shape[1])
frame=frame[int(heigh*2/3):heigh,0:width]
frame = Image.fromarray(np.uint8(frame))   

temp = max(frame.size)
mask = Image.new('RGB', (temp, temp))
NewWidth=frame.size[0]
NewHeigh=frame.size[1]
mask.paste(frame, (0, NewWidth-NewHeigh))

img = mask.resize(size)
img_data=transform(img).cuda()
img_data=torch.unsqueeze(img_data,dim=0)

float_model = UNet().to('cuda')

traced_model = torch.jit.trace(float_model, torch.randn(1, 3, 256, 256))
# 创建一个具有量化支持的模型
quantized_model = torch.quantization.quantize_dynamic(
    traced_model,  # 浮点模型
    {torch.nn.Conv2d},  # 选择需要量化的层
    dtype=torch.qfloat16  # 定点类型（默认为torch.qint8）
)

quantized_model.eval()

time1=time.time()
##开始预测
out = quantized_model(img_data)
# out=net(img_data)
out=torch.argmax(out,dim=1)
out=torch.squeeze(out,dim=0)
out=out.unsqueeze(dim=0)
out=(out).permute((1,2,0)).cpu().detach().numpy()
out=out*255.0
cv2.imwrite('./cat_superres_with_ort.jpg',out)