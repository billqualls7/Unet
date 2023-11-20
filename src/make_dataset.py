'''
Author: Wuyao 1955416359@qq.com
Date: 2023-04-28 09:08:49
LastEditors: Wuyao 1955416359@qq.com
LastEditTime: 2023-07-08 12:04:52
FilePath: \wyUnet\mysrc\make_dataset.py
Description: 制作数据集
'''
import os
import random
import shutil
from tqdm import tqdm
import time
# 设置原始数据集所在的目录和新的训练/验证/测试集目录
original_dataset_dir = r'E:\Code\UnetV2\originalDataset\image'                           #图片
original_label_dir = r'E:\Code\UnetV2\originalDataset\label'                      #标签
base_image_dir = r'E:\Code\UnetV2\newDataset\image'
base_label_dir = r'E:\Code\UnetV2\newDataset\label'                                #新的数据集目录
train_img_save_dir = r'E:\Code\UnetV2\trainDataset\JPEGImages'                     #直接将图片保存的训练路径下面
train_label_save_dir = r'E:\Code\UnetV2\trainDataset\SegmentationClass'             #直接将标签保存的训练路径下面
train_image_dir = os.path.join(base_image_dir, 'train')                                  #训练集
train_label_dir = os.path.join(base_label_dir, 'train')  
validation_image_dir = os.path.join(base_image_dir, 'validation')                        #验证集
validation_label_dir = os.path.join(base_label_dir, 'validation')  
test_image_dir = os.path.join(base_image_dir, 'test')                                    #测试集
test_label_dir = os.path.join(base_label_dir, 'test')  
# 创建新的目录结构
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(validation_image_dir, exist_ok=True)
os.makedirs(validation_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 将数据集文件名列表进行随机排序
filenames = os.listdir(original_dataset_dir)
random.shuffle(filenames)
# 将数据集按照7:2:1的比例分成训练集、验证集和测试集
train_size = int(0.7 * len(filenames))
val_size = int(0.2 * len(filenames))
test_size = len(filenames) - train_size - val_size

train_filenames = filenames[:train_size]
val_filenames = filenames[train_size:train_size+val_size]
test_filenames = filenames[train_size+val_size:]

# 创建进度条对象
pbar = tqdm(total=len(filenames))

# 将文件拷贝到新的目录中
for filename in train_filenames:
    src = os.path.join(original_dataset_dir, filename)
    src_lable = os.path.join(original_label_dir, filename)
    dst_img = os.path.join(train_img_save_dir, filename)
    dst_lable = os.path.join(train_label_save_dir, filename)
    shutil.copyfile(src, dst_img)
    shutil.copyfile(src_lable, dst_lable)
    pbar.update(1)

for filename in val_filenames:
    src = os.path.join(original_dataset_dir, filename)
    src_lable = os.path.join(original_label_dir, filename)
    dst_img = os.path.join(validation_image_dir, filename)
    dst_lable = os.path.join(validation_label_dir, filename)
    shutil.copyfile(src, dst_img)
    shutil.copyfile(src_lable, dst_lable)
    pbar.update(1)

for filename in test_filenames:
    src = os.path.join(original_dataset_dir, filename)
    src_lable = os.path.join(original_label_dir, filename)
    dst_img = os.path.join(test_image_dir, filename)
    dst_lable = os.path.join(test_label_dir, filename)
    shutil.copyfile(src, dst_img)
    shutil.copyfile(src_lable, dst_lable)
    pbar.update(1)


# 关闭进度条
pbar.close()