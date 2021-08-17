import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16

input_onnx_model = './onnx2ncnn/model.onnx'
output_onnx_model = './onnx2ncnn/model_fp16.onnx'

onnx_model = onnxmltools.utils.load_model(input_onnx_model)
onnx_model = convert_float_to_float16(onnx_model)
onnxmltools.utils.save_model(onnx_model, output_onnx_model)