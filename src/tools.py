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
sys.path.append("./unets")
from nets import read_yaml
from torch import nn, optim

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

def save_to_excel(var1,path):
    # 创建一个DataFrame对象
    df = pd.DataFrame({'loss': var1})
    
    # 将DataFrame保存到Excel文件
    excel_path= os.path.join(path,"train.xlsx")
    df.to_excel(excel_path, index=False, float_format='%.16f')


def train_print(min_loss_weight_path, prev_loss, min_loss_round,execution_time):
    print('\n------------------------------------------------------------------')
    print(f'\033[92mSave model successfully at {min_loss_weight_path}\033[0m')
    print(f'\033[92mmini_loss: {prev_loss}\033[0m')
    print(f'\033[92mmin_loss_round: {min_loss_round}\033[0m')
    print('\033[92m' + "Execution time: " + str(execution_time) + " minutes" + '\033[0m')
    print('------------------------------------------------------------------')

def mkdirr(save_dir, train = False):

    exp_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d)) and d.startswith('exp')]
    num_exp_dirs = len(exp_dirs)
    save_path = os.path.join(save_dir,f'exp{num_exp_dirs}')
    train_result_path=''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'\033[95m{current_time} save path:{save_path}\033[0m')
        train_result_path = os.path.join(save_path,"train_result")
        if train :os.makedirs(train_result_path)
    return save_path, train_result_path

def draw(train_epch,train_loss_list,weight_path):
    epochs = list(range(1, train_epch + 1))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(weight_path+'/train_loss.png')  # 保存图片
    plt.show()

def check(train_epch):
    if train_epch <150:
        print('\033[91m' + "Error: train_epoch is not enough. Exiting program." + '\033[0m')
        sys.exit()

def ReadData(data_path,max_batch_size):
    data_loader = DataLoader(MyDataset(data_path), batch_size=max_batch_size, shuffle=True)
    total_trainimgs = len(data_loader.dataset)
    print(f'\033[95mtrain_imgs:{total_trainimgs}\033[0m')
    return data_loader

def InitModel(yamlpath):
    net, data_path,  train_epch, max_batch_size, train_lr, train_wd, model = read_yaml(yamlpath)
    data_loader = ReadData(data_path,max_batch_size)
    save_dir = 'params'  # 保存模型的根文件夹路径
    weight_path, train_result_path = mkdirr(save_dir,train = True)
    check(train_epch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[95mdevice:{device}\033[0m')
    net = net.to(device)
    opt = optim.Adam(net.parameters(),lr = train_lr)
    


    return net, opt, train_epch, data_loader, weight_path, train_result_path, model



    

if __name__ == '__main__':
    InitModel('F:/Code/UnetV3/cofig/train.yaml')