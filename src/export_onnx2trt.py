# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-11-25 21:12:37
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-11-25 21:15:39
# FilePath: \UnetV3\src\export_onnx2trt.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
 
# 加载ONNX文件
onnx_file_path = 'run\onnx\exp11\model.onnx'
engine_file_path = 'run/trt/model_tesfp16.trt'
 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1)
parser = trt.OnnxParser(network, TRT_LOGGER)
 
# 解析ONNX文件
with open(onnx_file_path, 'rb') as f:
    data = f.read()
    parser.parse(data)
 
# 构建TensorRT引擎
builder_config = builder.create_builder_config()
builder_config.max_workspace_size = 4*(1 << 30)
builder_config.set_flag(trt.BuilderFlag.FP16) 
engine = builder.build_engine(network, builder_config)
 
# 构建TensorRT引擎
# builder_config = builder.create_builder_config()
# builder_config.max_workspace_size = 1 << 30
# # builder_config.max_batch_size = 1  # 设置最大批量大小
# builder_config.set_flag(trt.BuilderFlag.FP16) 
# # builder_config.set_flag(trt.BuilderFlag.INT8) 
# engine = builder.build_engine(network, builder_config)
 
 
 
 
# 保存TensorRT引擎到文件
with open(engine_file_path, 'wb') as f:
    f.write(engine.serialize())
 