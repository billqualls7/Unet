# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-11-01 22:00:15
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-11-21 21:36:16
# FilePath: \UnetV3\unets\nets.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''




from unet_Conv_Block import *
from unet_DoubleConv import *
from unet_Vgg import *
from unetv3 import *
from unetv3_2 import *
from unetv4 import *
import yaml
import sys

def read_yaml(file_path):
    try:
        with open(file_path, 'r',encoding='utf-8') as file:
            config = yaml.safe_load(file)
            # 将 YAML 转换为 Python 字典
            data_path = config['data_path']
            train_epch = config['train_epch']
            max_batch_size = config['max_batch_size']
            nc = config['num_classes'] + 1
            train_lr = config['train_lr']
            train_wd = config['train_wd']
            model = config['model']
            # net_list = ['UNetCB', 'UNetDC', 'VGG16UNet', 'UNetV3_2', 'UNetV3']
            net_dic = {'UNetCB': 'UNetCB(num_classes = nc)',
            'UNetDC': 'UNetDC(num_classes = nc)',
            'VGG16UNet': 'VGG16UNet(num_classes = nc)',
            'UNetV3_2': 'UNetV3_2(out_channels = nc)',
            'UNetV3': 'UNetV3(out_channels = nc)',
            'UNetV4': 'UNetV4(out_channels = nc)',

            }
            if model not in net_dic:
                print('\033[91m' + "Error Net. Please check your yaml. \n"+'path: '+file_path + '\033[0m')
                sys.exit()
            else:
                net = eval(net_dic[model])
                print('------------------------------------------------------------------')
                print('data_path:', data_path)
                print('train_epch:', train_epch)
                print('max_batch_size:', max_batch_size)
                print('num_classes:', nc)
                print('train_lr:', train_lr)
                print('train_wd:', train_wd)
                print('model:', model)
                print('------------------------------------------------------------------')
                return net, data_path,  train_epch, max_batch_size, train_lr, train_wd, model
        # return data
    except FileNotFoundError:
         print("配置文件不存在")

    except yaml.YAMLError:
        print("配置文件格式错误")

