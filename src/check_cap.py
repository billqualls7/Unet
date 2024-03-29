'''
Date: 2024-02-01 22:17:55
LastEditTime: 2024-02-03 15:00:25
FilePath: /tls_visualtask/Unet-3.4/src/check_cap.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import time
from tqdm import tqdm

class Camera:

    def __init__(self, device):
        self.device = device
    
    def checkparam(self):

        cap = cv2.VideoCapture(self.device)  

        if not cap.isOpened():
            print(f"Error: Couldn't open video capture device: {self.device}.")
        else:
            start_time = time.time()

            # 读取50帧作为测试
            for _ in tqdm(range(100), desc="Reading frames"):
                ret, frame = cap.read()
                if not ret:
                    break


            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 100 / elapsed_time
            ret, frame = cap.read()
            dimensions = frame.shape
            print(f"Open video capture device: {self.device}.")
            if len(dimensions) == 3:
                height, width, channels = dimensions
                print(f"Measured shape: {frame.shape} ")
                print(f"Measured Height = {height}, Width = {width}")
                print("Measured Color Channel Order: BGR") 
            else:
                height, width = dimensions
                print(f"Measured shape: {frame.shape} ")
                print(f"Measured Height = {height}, Width = {width}")
                print("This is a grayscale image.")
                
            print(f"Measured Frame Rate: {int(fps)} FPS")
            

        # 释放摄像头资源
        cap.release()

    def _resize(self, size = (240, 320)):
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[1])



if __name__ == "__main__":
    cap = Camera(0)
    cap.checkparam()
    cap._resize()
    _, frame = cap.cap.read()
    print(frame.shape)
