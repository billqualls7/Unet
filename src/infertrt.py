# 导入torch、tensorrt、cv2和numpy模块
import torch
import tensorrt as trt
import cv2
import numpy as np
# 导入cuda模块
import pycuda.autoinit
import pycuda.driver as cuda

# 定义你的模型文件路径、图片文件路径和输入大小
# model_path = 'run/trt/exp0/model.engine'
model_path = 'run/trt/model_tesfp16.trt'
image_path = 'demo/002370.png'
input_size = (320, 240)

# 定义一个函数，用来为TensorRT引擎分配输入和输出缓冲区
def allocate_buffers(engine, is_explicit_batch=False, input_shape=None):
    inputs = []
    outputs = []
    bindings = []
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem
    for binding in engine:
        dims = engine.get_binding_shape(binding)
        if dims[-1] == -1: # 如果是动态输入，需要指定输入的形状
            assert(input_shape is not None)
            dims[-2],dims[-1] = input_shape
        size = trt.volume(dims) * engine.max_batch_size # 计算缓冲区的大小
        dtype = trt.nptype(engine.get_binding_dtype(binding)) # 获取缓冲区的数据类型
        # 分配主机和设备内存
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # 把设备内存添加到绑定列表
        bindings.append(int(device_mem))
        # 判断是输入还是输出，并添加到相应的列表
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings

# 定义一个函数，用来执行推理
def do_inference_v2(context, bindings, inputs, outputs, stream, batch_size=1):
    # 将输入数据从主机复制到设备
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # 执行推理
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # 将输出数据从设备复制到主机
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # 同步流
    stream.synchronize()
    # 返回输出结果
    return [out.host for out in outputs]

# 创建一个TensorRT日志对象
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 加载你的.engine文件
with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 创建一个执行上下文和一个输入输出缓冲区
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine, input_shape=input_size)

# 读取你的图片文件
img = cv2.imread(image_path)
# 调整图片的大小和颜色通道
img = cv2.resize(img, input_size)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 转换图片的数据类型和范围
img = img.astype(np.float32)
img = img / 255.0
# 调整图片的维度和顺序
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# 把图片数据复制到输入缓冲区
np.copyto(inputs[0].host, img.ravel())

# 执行推理
trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

# 获取输出结果
output = trt_outputs[0].reshape((1, 1000))

# 打印输出结果
print(output)
