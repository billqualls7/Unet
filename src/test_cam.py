# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-07-12 21:19:10
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-07-12 21:21:19
# FilePath: \UnetV2\src\test_cam.py
# Description: 测试OPENCV GPU CPU读取图像的速度
# '''

import cv2
import time

def main():
    # 打开摄像头
    capture = cv2.VideoCapture(0)

    # 测试GPU加速
    start_time = time.time()
    # test_gpu(capture)
    gpu_time = time.time() - start_time

    # 打开摄像头
    capture = cv2.VideoCapture(0)

    # 测试CPU
    start_time = time.time()
    test_cpu(capture)
    cpu_time = time.time() - start_time

    print(f"GPU加速时间：{gpu_time:.4f} 秒")
    print(f"CPU时间：{cpu_time:.4f} 秒")

def test_gpu(capture):
    # 读取视频帧
    ret, frame = capture.read()

    if not ret:
        print("无法读取视频帧")
        return

    # 创建GPU加速的图像对象
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # 在GPU上进行图像处理（此处仅进行BGR到灰度的转换作为示例）
    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

    # 将处理后的图像从GPU下载到CPU内存
    gray = gpu_gray.download()

    # 显示图像
    cv2.imshow("Video (GPU)", gray)
    cv2.waitKey(0)

    # 释放资源
    capture.release()
    cv2.destroyAllWindows()

def test_cpu(capture):
    # 读取视频帧
    ret, frame = capture.read()

    if not ret:
        print("无法读取视频帧")
        return

    # 进行图像处理（此处仅进行BGR到灰度的转换作为示例）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 显示图像
    cv2.imshow("Video (CPU)", gray)
    cv2.waitKey(0)

    # 释放资源
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
