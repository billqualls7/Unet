import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
 
import numpy as np
import time
import cv2
from PIL import Image
from pprint import pprint

TRT_LOGGER = trt.Logger()
 
def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s


def get_img_np_nchw(image):
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (256, 256))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    img_np = (img_np - mean) / std
    img_np = img_np.transpose((2, 0, 1))
    img_np_nchw = np.expand_dims(img_np, axis=0)
    return img_np_nchw
 
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        super(HostDeviceMem, self).__init__()
        self.host = host_mem
        self.device = device_mem
 
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
    def __repr__(self):
        return self.__str__()
        
 
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()  # pycuda 操作缓冲区
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
 
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # 分配内存
        bindings.append(int(device_mem))
 
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
 
def get_engine(engine_file_path=""):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
 
 
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs] # 将输入放入device
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle) # 执行模型推理
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs] # 将预测结果从缓冲区取出
    stream.synchronize()    # 线程同步
    return [out.host for out in outputs]
 
def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs
 
def landmark_detection(image_path):
    trt_engine_path = './params/Unetv2fp16.trt'
 
    engine = get_engine(trt_engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
 
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 256))
    img_np_nchw = get_img_np_nchw(image)

    img_np_nchw = img_np_nchw.astype(dtype=np.float32)
    
    inputs[0].host = img_np_nchw.reshape(-1)
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t2 = time.time()
    print('used time: ', 1/(t2-t1))
 
    # shape_of_output = np.array((1, 6, 256, 256), dtype=np.float32)
    shape_of_output = (1,6,256,256)
    # print(shape_of_output)
    # print(trt_outputs[0])
   
    landmarks = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    # print(landmarks)
    # result = softmax(landmarks)
    landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
    
    height, width = image.shape[:2]
    pred_landmark = landmarks[0] * [height, width]

   
    # score, index = np.max(result, axis=1), np.argmax(result, axis=1)
    # print(score[0], index[0])
 
    for (x, y) in pred_landmark.astype(np.int32):
        cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
 
    # cv2.imshow('landmarks', image)
    # cv2.waitKey(0)
 
    return pred_landmark
 
if __name__ == '__main__':
    image_path = 'E:/Code/UnetV2/newDataset/image/test/002430.png'
    landmarks = landmark_detection(image_path)
 