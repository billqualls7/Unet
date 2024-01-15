# -*- coding:utf-8 _*-
from collections import OrderedDict
import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

class UNetV3_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, init_features=8):
        super(UNetV3_2, self).__init__()
        # print(out_channels)
        features = init_features
        # 编码
        self.encoder1 = UNetV3_2._block3(in_channels, features, name="enc1")

        self.encoder2 = UNetV3_2._block3(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNetV3_2._block3(features * 2, features * 2, name="enc3")
        self.dop3 = nn.Dropout(0.2)

        self.encoder4 = UNetV3_2._block3(features * 2, features * 4, name="enc4")
        self.dop4 = nn.Dropout(0.2)

        self.encoder5 = UNetV3_2._block3(features * 4, features * 4, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dop5 = nn.Dropout(0.2)

        self.encoder6 = UNetV3_2._block3(features * 4, features * 8, name="enc6")
        self.dop6 = nn.Dropout(0.2)

        self.encoder7 = UNetV3_2._block3(features * 8, features * 8, name="enc7")
        self.dop7 = nn.Dropout(0.2)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码
        self.decoder10 = UNetV3_2._Tblock2(features * 8, features * 8, name="dec10")

        self.decoder9 = UNetV3_2._Tblock3(features * 8, features * 8, name="dec9")
        self.tdop9 = nn.Dropout(0.2)

        self.decoder8 = UNetV3_2._Tblock3(features * 8, features * 8, name="dec8")
        self.tdop8 = nn.Dropout(0.2)

        self.decoder7 = UNetV3_2._Tblock2(features * 8, features * 8, name="dec7")

        self.decoder6 = UNetV3_2._Tblock3(features * 8, features * 4, name="dec6")
        self.tdop6 = nn.Dropout(0.2)

        self.decoder5 = UNetV3_2._Tblock3(features * 4, features * 4, name="dec5")
        self.tdop5 = nn.Dropout(0.2)

        self.decoder4 = UNetV3_2._Tblock3(features * 4, features * 2, name="dec4")
        self.tdop4 = nn.Dropout(0.2)

        self.decoder3 = UNetV3_2._Tblock2(features * 2, features * 2, name="dec3")

        self.decoder2 = UNetV3_2._block3(features * 2, features, name="dec2")

        # self.decoder1 = UNetV3_2._block3(features, out_channels, name="dec1")
        self.decoder1 = nn.Conv2d(features, out_channels, kernel_size=1, padding=0, bias=True)
        self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, mean=0, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # x=torch.randn(1,3,256,256)
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
        # print("11111111111111111111111111111")
        # print(dec9.shape)
        
        return out

    @staticmethod
    def _block3(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 用字典的形式进行网络定义，字典key即为网络每一层的名称
                [
                    (
                        name + "conv3",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def _Tblock3(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 用字典的形式进行网络定义，字典key即为网络每一层的名称
                [
                    (
                        name + "Tconv3",
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )

    @staticmethod
    def _Tblock2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 用字典的形式进行网络定义，字典key即为网络每一层的名称
                [
                    (name + "up",
                     nn.Upsample(
                         scale_factor=2,
                         mode='bilinear',
                         align_corners=True)
                     ),
                    (
                        name + "Tconv2",
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            # padding_mode='reflect',
                            bias=False,
                        )
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    net=UNetV3_2()
    net(x)
    print("--------")
    print(net(x).shape)
    print("--------")
