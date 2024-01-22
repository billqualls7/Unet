<!--
 * @Author: Wuyao 1955416359@qq.com
 * @Date: 2023-04-28 13:53:57
 * @LastEditors: Wuyao 1955416359@qq.com
 * @LastEditTime: 2023-11-25 19:35:04
 * @FilePath: \UnetV3\readme.md
 * @Description:readme
-->

# TLS-Unet

- src\make_dataset.py 划分数据集  
- 使用make_dataset.py可以将已经打完标签的数据集按照7：2：1的比例划分为训练集、验证集、测试集  
- 训练模型的时候要保证训练集的图片和标签的文件夹在一个目录下面，并且名字为JPEGImages和-JPEGImages  

```txt
例如：E:\Code\wyUnet\data\JPEGImages  E:\Code\wyUnet\data\SegmentationClass  

在train.py的如下代码中进行修改  

data_path = 'E:/Code/wyUnet/data'      
```



- [x] 请到src\utils.py size=(320, 240)修改模型输入输出大小即图片的H*W，后续集成到yaml文件中（v3.4.1已修复）

    

1. originalDataset  ：存储原始数据集及其标签

   ```python
   globalpath = "F:/Code/UnetV3/dataset/"
   orig = globalpath+"orignialDataset/" 	## 原始数据集
   train = globalpath+"trainDataset/"		## 训练集
   val = globalpath+"valDataset/"   		## 验证集
   test = globalpath+"testDataset/"		## 测试集
   ```

2. src\train.py  ：训练  终端进入src目录下

```bash
终端执行------------
F:\Code\UnetV3\src>python train.py -h
终端输出------------
usage: ./train.py [-h] [--yamlpath] [--onnx] [--val]

Unet train.py

optional arguments:
  -h, --help   show this help message and exit
  --yamlpath   train params default:cofig/train.yaml
  --onnx       creat onnx ? True or False default=False
  --val        use val ? True or False default=True

  
终端执行------------ 选择生成ONNX模型 
python train.py --onnx True
```

3. src\test.py ： 测试  （弃用）

4. src\test_ValDataset.py  ：使用测试test集数据跑模型（弃用）

5. 请使用板载最新版推理脚本

6. src/inferonnx.py 模型转为ONNX后，可用此脚本进行推理

7. ModuleNotFoundError: No module named 'nets' 出现类似情况，请在报错位置使用全局路径




### wyunetv2.1  
    优化了车道线计算角度的逻辑
    网络参数更改，可转换为TensorRT格式文件

### wyunetv2.2  
    修复已知BUG
    优化训练过程，增加训练可视化，无需指定权重保存路径，训练结果会保存在params/exp文件夹中

### wyunetv2.3  
    1.优化代码，模型会将损失值作为指标，模型训练结束之后会保存最后一次训练结束的模型和所有模型中损失值最小的模型
    2.训练结果会保存在params/exp文件夹中 
    3.epx文件夹中包括损失函数曲线图、每一轮损失值的excel表格、训练过程的对比图在params\exp\train_result 
    4.对比图左侧是标签，右侧是训练结果
    5.增加tools.py 提高代码规范性
    6.src\test_ValDataset.py 优化代码 只检测到单条线的时候会按照上次检测的结果去做曲线拟合
    6.1 nwindows: 窗门的数量。该参数决定了在图像中使用多少个窗口来搜索车道线。可以根据道路的宽度和车道线的曲率来调整该参数。如果道路比较宽或者车道线比较弯曲，可能需要增加窗口的数量
    6.2 margin: 窗口的宽度。该参数决定了窗口在x轴上的宽度，用于确定左右车道线的搜索范围。可以根据车道线的宽度来调整该参数。如果车道线较宽，可能需要增加窗口的宽度
    6.4 minpix: 每个窗口至少需要的像素点数量，该参数决定了窗口内的像素点数目，用于判断该窗口内是否存在车道线。可以根据图像的分辨率和车道线的清晰度来调整该参数。如果图像分辨率较高或者车道线较清晰，可能需要增加该参数的值。
    6.4 minLane: 最小的车道线宽度。该参数用于确定车道线的最小宽度，用于过滤掉宽度较小的噪声。可以根据道路的宽度来调整该参数。如果道路较窄，可能需要减小该参数的值
    6.5 增加窗口的数量和宽度可以提高车道线检测的准确性，因为更多的窗口和更宽的窗口可以覆盖更大的搜索范围，减小了漏检和误检的概率。同时，增加每个窗口至少需要的像素点数量和最小车道线宽度可以过滤掉较小的噪声，提升车道线检测的稳定性

## wyunet-v3  

    wyunet网络优化  
    1.在原来网络的基础上编码器和解码器部分使用深度可分离卷积。
    深度可分离卷积通过先进行深度卷积，再进行逐点卷积，从而减少参数量和计算量。
    2.在编码器部分，输入经过一次传统卷积操作后，再使用深度可分离卷积。
    3.在解码器部分，先进行上采样操作，然后将上采样结果与编码器部分的输出进行拼接，再使用深度可分离卷积。  
    4.在initialize_weights方法,使用了init.xavier_uniform_对卷积层的权重进行Xavier均匀初始化, 并使用init.normal_对逐点卷积的权重进行正态分布初始化。同时,对BatchNorm层的权重也进行了初始化。可以根据需要调整mean和std参数来设置不同的均值和标准差。  

### wyunet-v3.2  2023-10-15 15:12:28

    1.取消v3网络中深度可分离卷积部分，采用分离卷积虽然可以减少参数量，但转为ONNX模型之后精度会下降。也可能是训练集样本不够导致。v3网络使用src\train_SeparableUNet.py训练，扩大数据集之后重新实验  
    2.支持多种unet网络结构，差异详见源码。使用不同网络结构进行训练，详见样例                    
        src\train_egeunet.py  src\train_SeparableUNet.py  src\train_unetv3_2.py  
    3.弃用src\testv3_ValDataset.py  
    4.优化src\export_unet2onnx.py  
        支持unet模型转换，使用时需导入对应网络结构和训练好的参数，如下所示
        from Unetv3_2 import *
        weights='params\exp5\min_loss.pt'   
        net=UNet(out_channels = num_classes)   #import wyUnet
    5.优化推理算法，详见src\inferonnx.py
        需要将训练好的模型转为ONNX格式，进行推理。请自行安装ONNX-GPU部署，在训练结束后，只使用了CPU来验证模型的可行性  
    6.请参照src\inferonnx.py在EPAI-car中重新部署，以及与ROS的通信  
    7.若速度无法满足要求，可使用onnx模型在嵌入式平台上使用C++语言和onnxruntime-gpu库对模型进行加速推理。对处理结果处理即拟合二阶曲线该部分算法需要移植到C++，后续会更新
    8.训练使用src\train_unetv3_2.py  

### wyunet-v3.3  2023-11-03 22:01:07  

    1. 增加congfig文件夹，训练参数从cofig\train.yaml进行修改  
    2. 所有网络结构源码存放到unets  
    3. unets\nets.py统一管理网络结构，并读取参数文件进行修改  
    4. 只保留一个训练脚本，网络模型选择从参数文件修改  
    5. 优化模型训练和导出过程，支持训练结束后自动生成pth模型与onnx模型;训练结束后会弹出模型损失图，手动关闭后会生成onnx模型，注意终端打印信息即可    
    6. 删除egeunet  
    7. 调试代码请注意各个源码中注释的代码块  
    8. 后续优化方向  
    8.1 增加训练集原图和标签对比功能，训练前检查数据集  
    8.2 针对训练中断情况进行恢复训练，避免重新训练浪费不必要的时间  

#### wyunet-v3.3.1  2023-11-23 21:54:59
    1. 修复onnx导出后推理精度下降问题  
    2. 新数据集损失值无法下降跟数据集标签有关  
    3. 新增v4代码，经测试性能不如v3_2  

#### wyunet-v3.3.2  2023-11-24 23:23:49
    1. 新增libtorch转换程序src\export_unet2trace.py
    2. 添加ncnn格式的模型，暂时没用

#### wyunet-v3.3.3 2024-01-15 22:10:18

1. 修复训练脚本bug
2. 优化训练模型命令规则
3. 补充：请到src\utils.py size=(320, 240)修改模型输入输出大小即图片的H*W，后续集成到yaml文件中
4. 添加src\summary.py文件，可在训练结束之后将pt文件导入打印 PyTorch 模型结构和参数数量

```python
print(summary(net,input_size))
```

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 8, 240, 320]             216
              ReLU-2          [-1, 8, 240, 320]               0
            Conv2d-3         [-1, 16, 240, 320]           1,152
              ReLU-4         [-1, 16, 240, 320]               0
         MaxPool2d-5         [-1, 16, 120, 160]               0
            Conv2d-6         [-1, 16, 120, 160]           2,304
              ReLU-7         [-1, 16, 120, 160]               0
           Dropout-8         [-1, 16, 120, 160]               0
            Conv2d-9         [-1, 32, 120, 160]           4,608
             ReLU-10         [-1, 32, 120, 160]               0
          Dropout-11         [-1, 32, 120, 160]               0
           Conv2d-12         [-1, 32, 120, 160]           9,216
             ReLU-13         [-1, 32, 120, 160]               0
          Dropout-14         [-1, 32, 120, 160]               0
        MaxPool2d-15           [-1, 32, 60, 80]               0
           Conv2d-16           [-1, 64, 60, 80]          18,432
             ReLU-17           [-1, 64, 60, 80]               0
          Dropout-18           [-1, 64, 60, 80]               0
           Conv2d-19           [-1, 64, 60, 80]          36,864
             ReLU-20           [-1, 64, 60, 80]               0
          Dropout-21           [-1, 64, 60, 80]               0
        MaxPool2d-22           [-1, 64, 30, 40]               0
         Upsample-23           [-1, 64, 60, 80]               0
  ConvTranspose2d-24           [-1, 64, 60, 80]          36,864
      BatchNorm2d-25           [-1, 64, 60, 80]             128
          Dropout-26           [-1, 64, 60, 80]               0
  ConvTranspose2d-27           [-1, 64, 60, 80]          36,864
             ReLU-28           [-1, 64, 60, 80]               0
      BatchNorm2d-29           [-1, 64, 60, 80]             128
          Dropout-30           [-1, 64, 60, 80]               0
  ConvTranspose2d-31           [-1, 64, 60, 80]          36,864
             ReLU-32           [-1, 64, 60, 80]               0
      BatchNorm2d-33           [-1, 64, 60, 80]             128
         Upsample-34         [-1, 64, 120, 160]               0
  ConvTranspose2d-35         [-1, 64, 120, 160]          36,864
      BatchNorm2d-36         [-1, 64, 120, 160]             128
          Dropout-37         [-1, 64, 120, 160]               0
  ConvTranspose2d-38         [-1, 32, 120, 160]          18,432
             ReLU-39         [-1, 32, 120, 160]               0
      BatchNorm2d-40         [-1, 32, 120, 160]              64
          Dropout-41         [-1, 32, 120, 160]               0
  ConvTranspose2d-42         [-1, 32, 120, 160]           9,216
             ReLU-43         [-1, 32, 120, 160]               0
      BatchNorm2d-44         [-1, 32, 120, 160]              64
          Dropout-45         [-1, 32, 120, 160]               0
  ConvTranspose2d-46         [-1, 16, 120, 160]           4,608
             ReLU-47         [-1, 16, 120, 160]               0
      BatchNorm2d-48         [-1, 16, 120, 160]              32
         Upsample-49         [-1, 16, 240, 320]               0
  ConvTranspose2d-50         [-1, 16, 240, 320]           2,304
      BatchNorm2d-51         [-1, 16, 240, 320]              32
           Conv2d-52          [-1, 8, 240, 320]           1,152
             ReLU-53          [-1, 8, 240, 320]               0
           Conv2d-54          [-1, 3, 240, 320]             216
             ReLU-55          [-1, 3, 240, 320]               0
================================================================
Total params: 256,880
Trainable params: 256,880
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.88
Forward/backward pass size (MB): 230.27
Params size (MB): 0.98
Estimated Total Size (MB): 232.13
----------------------------------------------------------------
```

规范代码书写，添加训练说明

```bash
终端执行------------
python train.py -h
终端输出------------
usage: ./train.py [-h] [--yamlpath] [--onnx]

Unet train.py

optional arguments:
  -h, --help   show this help message and exit
  --yamlpath   train params
  --onnx       creat onnx ? True or False default=False
  
终端执行------------ 选择生成ONNX模型 
python train.py --onnx True

终端输出------------

------------------------------------------------------------------
model:             UNetV3_2
data_path:         F:/Code/UnetV3/trainDataset
train_epch:        4
max_batch_size:    16
num_classes:       6
train_lr:          0.01
train_wd:          0.0
img_size_H*W:      (320, 240)
------------------------------------------------------------------     
2024-01-16 22:25:59
save path:     ../params\exp11
train_imgs:         122
device:             cuda
0%|                                           | 0/8 [00:00<?, ?it/s]0-0-train_loss===>>1.9120261669158936
 12%|████▍                              | 1/8 [00:01<00:10,  1.48s/it]0-1-train_loss===>>1.8240903615951538
 25%|████████▊                          | 2/8 [00:02<00:05,  1.09it/s]0-2-train_loss===>>1.695765733718872
 38%|█████████████▏                     | 3/8 [00:02<00:03,  1.36it/s]0
 ....
 ....
 .....
------------------------------------------------------------------
Trans model successfully at ../params\exp11\model.onnx
------------------------------------------------------------------     
------------------------------------------------------------------
Save model successfully at    ../params\exp11\UNetV3_2_min_loss.pt
mini_loss:                    0.039560265839099884
min_loss_round:               3
Execution time:               0.29 minutes
------------------------------------------------------------------ 

```

### wyunet-v3.4 2024-01-19

优化src/make_dataset.py数据集划分代码结构

修改训练逻辑，增加验证集，

请到val文件夹中查看训练效果

> （在每个训练周期结束后，通过比较验证集的损失，决定是否需要降低学习率。如果验证集的损失没有下降，则将学习率减小到原来的10%。**数据集样本过少，这点还需验证是否生效**       这块代码有点问题 之后再改）
>
> 优化数据集存放结构
>
> 增加混淆矩阵及其相关性能指标可视化模块，详见src\tools.py SegmentationMetric
>
> [【语义分割】评价指标：PA、CPA、MPA、IoU、MIoU详细总结和代码实现_语义分割mpa计算公式-CSDN博客](https://blog.csdn.net/smallworldxyl/article/details/121570419?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-5-121570419-blog-127939004.235^v40^pc_relevant_anti_vip_base&spm=1001.2101.3001.4242.4&utm_relevant_index=8)
>
> acc曲线存疑，后续有足够的数据集再验证这部分的正确性
>

#### wyunet-v3.4.1 2024-01-22

增加验证集参数，默认开启，训练结束前会对验证集进行推理并且将图片保存到和模型同一目录下

```python
  parser.add_argument('--val', type=bool, metavar='', default=True, help='use val ? True or False default=False')
```

增加fire模块，详见SqueezeNet

```latex
@article{iandola2016squeezenet,
  title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size},
  author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
  journal={arXiv preprint arXiv:1602.07360},
  year={2016}
}
```

优化网络结构，即unets\unet_Fire.py，该网络需要增加训练轮次，最少200次以上

根据summary.py程序计算，该网络结构参数量减少约13%

|   Model   | Estimated Total Size (MB): |
| :-------: | :------------------------: |
| UNetV3_2  |           232.13           |
| UNet_Fire |           203.56           |

仅用CPU推理 （pytorch原生推理）

|   Model   | FPS  |
| :-------: | :--: |
| UNetV3_2  | 13.5 |
| UNet_Fire | 17.8 |

- [ ] 更换数据集重新训练一下，UNet_Fire在识别左车道线时收敛比较慢，感觉像是数据集的问题



# 以下思路可提供参考 2023-07-22 15:02:48更新

    1. 由于git库的删除，v2.1中的提及到的功能无法实现，通过历史找回了wyunet在fp16精度下的trt模型，以及测试脚本src\test_tensorrt.py 通过测试发现在加速之后帧率在3060Ti G6X上面能够跑到500帧/秒，但精度严重缺失，应该是在转换过程中出现了问题，未来可以在此基础上进一步研究。需要注意的是该版本训练生成的模型无法转为trt模型，需要对src\train.py进行更改，将整个网络结构及其参数进行保存，该功能和模型转换部分的代码已经丢失。理论上来说，v2.1的train.py训练之后保存的模型，对其重新构建推理器，使其不再调用src\Unet.py，在嵌入式平台的推理应该会有加快。
    2. model.onnx为wyunet的onnx格式模型，该模型同样存在精度严重缺失，应该是在转换过程中出现了问题，未来可以在此基础上进一步研究。思路为使用onnx模型在嵌入式平台上使用C++语言和onnxruntime-gpu库对模型进行加速推理。相关代码为src\export_unet.py src\detect_onnx.py 
    3. src\demo.py  将卷积函数进行量化，减少计算量实现模型的快速推理，该功能未能实现
    4. 针对比赛时的车道线检测，只要求了车道线，没有对车道线的类别进行要求，目前的推理器将所有非背景的标签统一进行二值化了，对于模型来说可以理解为一种标签，所以可以尝试制作数据集时减少标签的类别，理论上模型减少了输出参数，是可以在推理时实现一定的加速的，当输出通道数较大时，会生成更多的特征图，导致计算量增加。同时，理论上来说，v2.1的train.py训练之后保存的模型，对其重新构建推理器，使其不再调用src\Unet.py，在嵌入式平台的推理应该会有加快。二者结合一下
    5. 对整个网络进行重构，使用libtorch-win-shared-with-deps-2.0.1+cu118.zip 提供的C++接口，将全部代码使用C++进行重构，并部署到嵌入式平台