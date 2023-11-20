# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-07-22 12:51:42
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-07-22 13:55:16
# FilePath: \UnetV2\src\Unetv3.py
# Description: 在原来网络的基础上编码器和解码器部分使用深度可分离卷积。
#             深度可分离卷积通过先进行深度卷积，再进行逐点卷积，从而减少参数量和计算量。
#             在编码器部分，输入经过一次传统卷积操作后，再使用深度可分离卷积。
#             在解码器部分，先进行上采样操作，然后将上采样结果与编码器部分的输出进行拼接，再使用深度可分离卷积。
#             在initialize_weights方法,使用了init.xavier_uniform_对卷积层的权重进行Xavier均匀初始化,
#             并使用init.normal_对逐点卷积的权重进行正态分布初始化。同时,对BatchNorm层的权重也进行了初始化。
#             可以根据需要调整mean和std参数来设置不同的均值和标准差。
# '''


from collections import OrderedDict
import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


class UNetV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=8):
        super(UNetV3, self).__init__()
        features = init_features

        # 编码
        self.encoder1 = self._block3(in_channels, features, name="enc1")

        self.encoder2 = self._depthwise_block3(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._depthwise_block3(features * 2, features * 2, name="enc3")
        self.dop3 = nn.Dropout(0.2)

        self.encoder4 = self._depthwise_block3(features * 2, features * 4, name="enc4")
        self.dop4 = nn.Dropout(0.2)

        self.encoder5 = self._depthwise_block3(features * 4, features * 4, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dop5 = nn.Dropout(0.2)

        self.encoder6 = self._depthwise_block3(features * 4, features * 8, name="enc6")
        self.dop6 = nn.Dropout(0.2)

        self.encoder7 = self._depthwise_block3(features * 8, features * 8, name="enc7")
        self.dop7 = nn.Dropout(0.2)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码
        self.decoder10 = self._depthwise_Tblock2(features * 8, features * 8, name="dec10")

        self.decoder9 = self._depthwise_Tblock3(features * 8, features * 8, name="dec9")
        self.tdop9 = nn.Dropout(0.2)

        self.decoder8 = self._depthwise_Tblock3(features * 8, features * 8, name="dec8")
        self.tdop8 = nn.Dropout(0.2)

        self.decoder7 = self._depthwise_Tblock2(features * 8, features * 8, name="dec7")

        self.decoder6 = self._depthwise_Tblock3(features * 8, features * 4, name="dec6")
        self.tdop6 = nn.Dropout(0.2)

        self.decoder5 = self._depthwise_Tblock3(features * 4, features * 4, name="dec5")
        self.tdop5 = nn.Dropout(0.2)

        self.decoder4 = self._depthwise_Tblock3(features * 4, features * 2, name="dec4")
        self.tdop4 = nn.Dropout(0.2)

        self.decoder3 = self._depthwise_Tblock2(features * 2, features * 2, name="dec3")

        self.decoder2 = self._depthwise_block3(features * 2, features, name="dec2")

        self.decoder1 = self._depthwise_block3(features, out_channels, name="dec1")
    
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, mean=0, std=0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
       
        enc2 = self.encoder2(enc1)
       
        enc3 = self.dop3(self.encoder3(self.pool2(enc2)))
        enc4 = self.dop4(self.encoder4(enc3))
        enc5 = self.dop5(self.encoder5(enc4))
        enc6 = self.dop6(self.encoder6(self.pool5(enc5)))
        enc7 = self.dop7(self.encoder7(enc6))

        enc = self.pool7(enc7)

        dec9 = self.tdop9(self.decoder10(enc))
        
        dec8 = self.tdop8(self.decoder9(dec9))
        
        dec7 = self.decoder8(dec8)
        dec6 = self.tdop6(self.decoder7(dec7))
        dec5 = self.tdop5(self.decoder6(dec6))
        dec4 = self.tdop4(self.decoder5(dec5))
        dec3 = self.decoder4(dec4)
        dec2 = self.decoder3(dec3)
        dec1 = self.decoder2(dec2)
        out = self.decoder1(dec1)
        
        return out

    @staticmethod
    def _block3(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _depthwise_block3(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _depthwise_Tblock3(in_channels, features, name):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features),
        )

    @staticmethod
    def _depthwise_Tblock2(in_channels, features, name):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
        )

if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    net=UNetV3()
    net(x)
    print("--------")
    print(net(x).shape)
    print("--------")
