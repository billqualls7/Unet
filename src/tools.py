# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-07-07 14:55:04
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-07-22 13:58:36
# FilePath: \UnetV2\src\tools.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''


import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import sys
from torch.utils.data import DataLoader
from data import *
import cv2
sys.path.append('../')
from  unets import *
from torch import nn, optim
import yaml


def find_line_fit(img, name = "default" ,nwindows=4, margin=100, minpix=100 , minLane = 100):
    # previous_left_fit = None
    # previous_right_fit = None
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
        # print(right_fit)
        
        return left_fit, right_fit
    except:
        print('\033[91m' +"error:" + name+ '\033[0m')
        # print(previous_right_fit)
        return [-100, -100, -100], [-100, -100, -100]
    
    # previous_left_fit = left_fit
    # previous_right_fit = right_fit
    # return left_fit, right_fit
    

    
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
    # cv2.line(cropped, (mid, 0), (mid,cropped.shape[0]), (0, 0, 255), thickness=10)
    # cv2.line(cropped,(bottom_x_left,bottom_y),(bottom_x_right,bottom_y),(255,0,0), thickness=10)
    # cv2.line(cropped, (mid, bottom_y), (int(bottom_x_left / 2 + bottom_x_right / 2), bottom_y), (0, 255, 0),
    #         thickness=10)
    
    # cv2.imwrite("result/img.jpg",cropped)
    # print(angle)
    angle = int(bottom_x_left / 2 + bottom_x_right / 2) - mid

    return angle

def load_image_as_array(image_path, size, alpha=1, beta=11):
    # 使用OpenCV加载图像
    frame = cv2.imread(image_path)

#---------------------------------------------------------------------------------------------------------
    # alpha = 1  # 对比度调整参数，大于1表示增加对比度，小于1表示降低对比度
    # beta = 10  # 亮度调整参数，每个像素点都加上该参数
#---------------------------------------------------------------------------------------------------------
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    height, width, _ = frame.shape

    black_region = np.zeros((height*2//3, width, 3), dtype=np.uint8) # 创建黑色区域
    frame[0:height*2//3, :] = black_region # 将上三分之二替换为黑色

    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = frame.resize(size)




    
    # plt.imshow(frame, cmap='gray')
    # plt.show()
    # array = np.array(frame).astype(np.float32)
    array = np.array(frame)
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # pil_image = Image.open(image_path)
    # pil_image = pil_image.resize(size).convert("RGB")
    # array = np.array(pil_image)

    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return array
def resize_image(image, size):
    # 获取原始图像的高度和宽度
    height, width = image.shape[:2]

    # 确定新的高度和宽度
    new_height, new_width = size

    # 计算缩放比例
    scale_x = new_width / width
    scale_y = new_height / height

    # 调整图像大小
    resized = cv2.resize(image, (new_width, new_height))

    return resized

# def load_image_as_array(image_path, size):
#     # 使用OpenCV加载图像
#     image = cv2.imread(image_path)

#     # 调整图像大小以适应模型输入要求
#     resized = resize_image(image, size)

#     # 将图像转换为浮点类型的NumPy数组
#     array = resized.astype(np.float32)

#     # 将通道顺序从BGR转换为RGB（如果需要）
#     array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

#     # 调整数组形状以适应模型输入
#     array = np.transpose(array, (2, 0, 1))
#     array = np.expand_dims(array, axis=0)

#     return array

def save_to_excel(name,var1,path):
    # 创建一个DataFrame对象
    df = pd.DataFrame({name: var1})
    
    # 将DataFrame保存到Excel文件
    excel_path= os.path.join(path,"train.xlsx")
    df.to_excel(excel_path, index=False, float_format='%.16f')


def train_print(min_loss_weight_path, prev_loss, min_loss_round,execution_time):
    print('------------------------------------------------------------------')
    print('%-30s%s' % ('Save model successfully at', '\033[92m{}\033[0m'.format(min_loss_weight_path)))
    print('%-30s\033[92m%s\033[0m' % ('mini_loss:', prev_loss))
    print('%-30s\033[92m%s\033[0m' % ('min_loss_round:', min_loss_round))
    print('%-30s\033[92m%s minutes\033[0m' % ('Execution time:', execution_time))
    print('------------------------------------------------------------------')

def mkdirr(save_dir, train = False):

    exp_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d)) and d.startswith('exp')]
    num_exp_dirs = len(exp_dirs)
    save_path = os.path.join(save_dir,f'exp{num_exp_dirs}')
    train_result_path=''
    val_path = ''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{current_time} \nsave path:     \033[92m{save_path}\033[0m')        
        train_result_path = os.path.join(save_path,"train_result")
        val_path = os.path.join(save_path,"val")
        if train :
            os.makedirs(train_result_path)
            os.makedirs(val_path)
    return save_path, train_result_path

def draw(name, epch,train_list,weight_path):
    epochs = list(range(1, epch + 1))
    plt.plot(epochs, train_list, label=name)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    plt.savefig(weight_path+'/' + name + '.png')  # 保存图片
    plt.show()

def check(train_epch):
    if train_epch <150:
        print('\033[91m' + "Error: train_epoch is not enough. Exiting program." + '\033[0m')
        sys.exit()

def ReadData(data_path,max_batch_size,imgsize):
    data_loader = DataLoader(MyDataset(data_path, size=imgsize), batch_size=max_batch_size, shuffle=True)

    return data_loader


class InitModel():
    def __init__(self, yamlpath):
        self.yamlpath = yamlpath
        self.train_data_path = ''
        self.val_data_path = ''
        self.train_epch = 0
        self.max_batch_size = 0
        self.nc = 0
        self.train_lr = 0
        self.train_wd = 0
        self.model = ''
        self.img_size = (320, 240)
        self.debug = False

        self.read_yaml()
        self.data_loader = ReadData(self.train_data_path,self.max_batch_size,self.img_size)
        self.data_loader_val = ReadData(self.val_data_path,1,self.img_size)

        if not self.debug:
            check(self.train_epch)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_trainimgs = len(self.data_loader.dataset)
        save_dir = '../params'  # 保存模型的根文件夹路径
        self.weight_path, self.train_result_path = mkdirr(save_dir,train = True)
        self.net = self.net.to(device)
        self.opt = optim.Adam(self.net.parameters(),lr = self.train_lr)
        print('train_imgs:         \033[92m{}\033[0m'.format(total_trainimgs))
        print('device:             \033[92m{}\033[0m'.format(device))


    def read_yaml(self):
        try:
            with open(self.yamlpath, 'r',encoding='utf-8') as file:
                config = yaml.safe_load(file)
                # 将 YAML 转换为 Python 字典
                self.train_data_path = config['train_data_path']
                self.val_data_path = config['val_data_path']
                self.train_epch = config['train_epch']
                self.max_batch_size = config['max_batch_size']
                self.nc = config['num_classes'] + 1
                self.train_lr = config['train_lr']
                self.train_wd = config['train_wd']
                self.model = config['model']
                self.debug = config['debug']
                # net_list = ['UNetCB', 'UNetDC', 'VGG16UNet', 'UNetV3_2', 'UNetV3']
                net_dic = {'UNetCB': 'UNetCB(num_classes = self.nc)',
                'UNetDC': 'UNetDC(num_classes = self.nc)',
                'VGG16UNet': 'VGG16UNet(num_classes = self.nc)',
                'UNetV3_2': 'UNetV3_2(out_channels = self.nc)',
                'UNetV3': 'UNetV3(out_channels = self.nc)',
                'UNetV4': 'UNetV4(out_channels = self.nc)',

                }
                self.img_size = (config["img_size"]["H"], config["img_size"]["W"])
                if self.model not in net_dic:
                    print('\033[91m' + "Error Net. Please check your yaml. \n"+'path: '+self.file_path + '\033[0m')
                    sys.exit()
                else:
                    self.net = eval(net_dic[self.model])
                    print('------------------------------------------------------------------')
                    print('debug:             \033[92m{}\033[0m'.format(self.debug))
                    print('model:             \033[92m{}\033[0m'.format(self.model))
                    print('train:             \033[92m{}\033[0m'.format(self.train_data_path))
                    print('train_epch:        \033[92m{}\033[0m'.format(self.train_epch))
                    print('max_batch_size:    \033[92m{}\033[0m'.format(self.max_batch_size))
                    print('num_classes:       \033[92m{}\033[0m'.format(self.nc))
                    print('train_lr:          \033[92m{}\033[0m'.format(self.train_lr))
                    print('train_wd:          \033[92m{}\033[0m'.format(self.train_wd))
                    print('img_size_H*W:      \033[92m{}\033[0m'.format(self.img_size))
                    print('------------------------------------------------------------------')
            # return data
        except FileNotFoundError:
            print("配置文件不存在")

        except yaml.YAMLError:
            print("配置文件格式错误")


# def InitModel(yamlpath):
#     net, data_path,\
#     train_epch, max_batch_size,\
#     train_lr, train_wd, model, imgsize = read_yaml(yamlpath)
#     data_loader = ReadData(data_path,max_batch_size,imgsize)
#     save_dir = '../params'  # 保存模型的根文件夹路径
#     weight_path, train_result_path = mkdirr(save_dir,train = True)
#     check(train_epch)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     total_trainimgs = len(data_loader.dataset)
#     print('train_imgs:         \033[92m{}\033[0m'.format(total_trainimgs))
#     print('device:             \033[92m{}\033[0m'.format(device))

#     net = net.to(device)
#     opt = optim.Adam(net.parameters(),lr = train_lr)
    


#     return net, opt, train_epch, data_loader, weight_path, train_result_path, model


import numpy as np
__all__ = ['SegmentationMetric']
 
"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
 
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU) # 求各类别IoU的平均
        return mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
    

if __name__ == '__main__':

    InitModel('F:/Code/UnetV3/cofig/train.yaml')
    # read_yaml('F:/Code/UnetV3/cofig/train.yaml')
    # metric = SegmentationMetric(6)
