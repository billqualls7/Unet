'''
Author: Wuyao
LastEditTime: 2024-02-03 17:00:05
FilePath: /tls_visualtask/Unet-3.4/src/test_tensorrt.py
'''



import os
import cv2
import numpy as np
import torch
import time
# from Unetv3 import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import tools
from torch.utils import mkldnn as mkldnn_utils
import unets
from torch2trt import TRTModule

# from egeunet import *
#'''
#description: 
#param {*} img
#param {*} name
#param {*} nwindows
#param {*} margin
#param {*} minpix
#param {*} minLane

#return {*}
#'''
def find_line_fit(img, name = "default" ,nwindows=4, margin=100, minpix=100 , minLane = 100):
    global previous_left_fit
    global previous_right_fit
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(img.shape[1]/2)
    # Set height of windows
    window_height = np.int32(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    h = [0,img.shape[0]]
    w = [0,img.shape[1]]
    leftx_current = w[0]
    rightx_current = w[1]
    # Step through the windows one by one
    for window in range(nwindows):
        start = h[1] - int(h[0] + (h[1] - h[0]) * window / nwindows)
        end = h[1] - int(h[0] + (h[1] - h[0]) * (window + 1) / nwindows)
        # print(start)
        
        # print(end)
        # print('///////////////////////')
        histogram = np.sum(img[end:start,w[0]:w[1]], axis=0)

        leftx_current = np.argmax(histogram[:midpoint]) if np.argmax(histogram[:midpoint]) > minLane else leftx_current
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint if np.argmax(histogram[midpoint:]) > minLane else rightx_current

        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # out_img = img
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        # (0,255,0), 2)
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        # (0,255,0), 2)
        
        # cv2.line(out_img,(leftx_current,0),(leftx_current,img.shape[1]),(255,0,0))
        # cv2.line(out_img, (rightx_current, 0), (rightx_current, img.shape[1]), (255, 0, 0))
        # cv2.line(out_img, (midpoint, 0), (midpoint, img.shape[1]), (255, 0, 0))
        
        # cv2.imshow("rec",out_img)
        # cv2.waitKey(0)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        print(right_fit)
        previous_left_fit = left_fit
        previous_right_fit = right_fit
        return left_fit, right_fit
    except:
        print('\033[91m' +"error:" + name+ '\033[0m')
        print(previous_right_fit)
        return previous_left_fit, previous_right_fit
    

    
def get_angle(img):

    cropped = img[180:256,0:256]   #  important  param
    # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray=cropped
    left, right = find_line_fit(gray)
    # print(left)
    # print(right)
    bottom_y = int(cropped.shape[0]/2)
    bottom_x_left = int(left[0] * (bottom_y ** 2) + left[1] * bottom_y + left[2])
    bottom_x_right = int(right[0] * (bottom_y ** 2) + right[1] * bottom_y + right[2])

    mid = 128
    #可视化
    cv2.line(cropped, (mid, 0), (mid,cropped.shape[0]), (0, 0, 255), thickness=10)
    cv2.line(cropped,(bottom_x_left,bottom_y),(bottom_x_right,bottom_y),(255,0,0), thickness=10)
    cv2.line(cropped, (mid, bottom_y), (int(bottom_x_left / 2 + bottom_x_right / 2), bottom_y), (0, 255, 0),
            thickness=10)
    
    # cv2.imwrite("result/img.jpg",cropped)
    # print(angle)
    angle = int(bottom_x_left / 2 + bottom_x_right / 2) - mid
    return angle




if __name__ == "__main__":
    time0 = time.time()
    save_dir = '../run/infer'  # 保存推理结果的根文件夹路径
    # weights=r'F:\Code\UnetV3\params\exp1\UNetV3_2_min_loss.pt'   
    weights = '../pre_model/UNetV3_2.engine'                   #权重路径+名称  UNetV3_2_min_loss
    _input='../demo'    #测试集路径
    # num_classes = 6 #标签数量
    previous_left_fit = [0, 0, 0]
    previous_right_fit = [0, 0, 0]
    # net=UNet(6).cuda()
    # net=unets.UNet_Fire(out_channels = 6).cuda()
    # net=unets.UNetV3_2(out_channels = 6)    #import wyUnet
    # net = EGEUNet(num_classes = num_classes).cuda() 
    # print(next(net.parameters()).device) 
    # size=(256, 256)
    net = TRTModule()
    image_path = '../dataset/trainDataset/JPEGImages/005670.png'


    size = (320, 240)
    filenames = os.listdir(_input)
    i=len(filenames)

    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        # net = torch.load(weights, map_location=torch.device('cpu'))
        # net.half()
        # net.eval()
        # net = mkldnn_utils.to_mkldnn(net)
        print('successfully')
        # save_path, _ = tools.mkdirr(save_dir)
        save_path = '../run/infer'
        for i in range(30):
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
            # img_data=transform(img)
            img_data=torch.unsqueeze(img_data,dim=0)
            # plt.imshow(img)
            # plt.show()
            # net.eval()
            time1=time.time()
            ##开始预测
            # _,out=net(img_data)
            # print(img_data.shape)
            # img_data = img_data.to_mkldnn()
            out=net(img_data)
            # out0 = out0[0]
            # out0 = out0.squeeze(1).cpu().detach().numpy()

            # out = out[0]
            
            # out = np.squeeze(out, axis=0)
            # plt.imshow(out, cmap = 'gray')
            out=torch.argmax(out,dim=1)
            out=torch.squeeze(out,dim=0)
            out=out.unsqueeze(dim=0)
            out=(out).permute((1,2,0)).cpu().detach().numpy()
            out=out*255.0
            
            save_path_out = os.path.join(save_path,'infer.png')
            cv2.imwrite(save_path_out,out)

            #计算偏差角度
            vtherror=get_angle(out)
            time2=time.time()
            # plt.imshow(out, cmap='gray')
            # plt.show()
            print('---------------------------')
            print("vtherror:",vtherror)
            print("fps:",1/(time2-time1))
            
        # time3 = time.time()
        # print("time:",(time3-time0))

    else:
        print('no loading')

   
   


