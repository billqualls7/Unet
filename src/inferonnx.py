# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-10-13 21:00:01
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2024-01-19 19:38:37
# FilePath: \UnetV3\src\inferonnx.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''





import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
from tools import *
from tools import load_image_as_array
from tools import get_angle





if __name__ == '__main__':
    previous_left_fit = [0, 0, 0]
    previous_right_fit = [0, 0, 0]
    image_path = '../demo/002370.png'
    # 'run\onnx\exp1\model.onnx'
    model_path = '../params/exp7/model.onnx'
    input_size = (320, 240)
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])  #需改成GPU

    
    input_data = load_image_as_array(image_path, input_size)
    input_data = input_data.astype(np.float32)

    time1 = time.time()
    output = session.run(None, {'input': input_data})[0]

    # print(output.shape) 
    output_array = output[0]
    output_array = np.transpose(output_array, (1, 2, 0))
    predicted_labels = np.argmax(output_array, axis=2)

    # 将类别映射为灰度值(0-255)
    gray_image = predicted_labels.astype(np.uint8) 
   
    vtherror=get_angle(gray_image)

    time2 = time.time() 
    plt.imshow(gray_image)
    plt.show()
    
    print("vtherror:",vtherror)
    # # print(time2-time1)
    print("fps:",1/(time2-time1))
