import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class UNetV4(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=8):
        super(UNetV4, self).__init__()
        features = init_features

        # Encoding
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

        # Decoding
        self.decoder10 = self._depthwise_Tblock2(features * 8, features * 8, name="dec10")
        self.decoder9 = self._depthwise_Tblock3(features * 8, features * 8, name="dec9")
        self.decoder8 = self._depthwise_Tblock3(features * 8, features * 8, name="dec8")
        self.decoder7 = self._depthwise_Tblock2(features * 8, features * 8, name="dec7")
        self.decoder6 = self._depthwise_Tblock3(features * 8, features * 4, name="dec6")
        self.decoder5 = self._depthwise_Tblock3(features * 4, features * 4, name="dec5")
        self.decoder4 = self._depthwise_Tblock3(features * 4, features * 2, name="dec4")
        self.decoder3 = self._depthwise_Tblock2(features * 2, features * 2, name="dec3")
        self.decoder2 = self._depthwise_block3(features * 2, features, name="dec2")
        # self.decoder1 = self._depthwise_block3(features, out_channels, name="dec1")
        self.decoder1 = nn.Conv2d(features, out_channels, kernel_size=1, padding=0, bias=True)


        # Residual blocks
        self.residual_block9 = self._residual_block(features * 8, features * 8)
        self.residual_block8 = self._residual_block(features * 8, features * 8)
        self.residual_block6 = self._residual_block(features * 8, features * 4)
        self.residual_block5 = self._residual_block(features * 4, features * 4)
        self.residual_block4 = self._residual_block(features * 4, features * 2)

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

        # Decoding with residual connections
        dec10 = self.decoder10(enc)
        dec9 = self.residual_block9(dec10) + self.decoder9(dec10)

        dec8 = self.residual_block8(dec9) + self.decoder8(dec9)
        dec7 = self.decoder7(dec8)
        dec6 = self.residual_block6(dec7) + self.decoder6(dec7)
        dec5 = self.residual_block5(dec6) + self.decoder5(dec6)
        dec4 = self.residual_block4(dec5) + self.decoder4(dec5)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)
        
        out = dec1

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
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
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
    def _residual_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
    
if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    net=UNetV4()
    net(x)
    print("--------")
    print(net(x).shape)
    print("--------")
