import os

precision = 'FP32'
model_name = 'lite'
onnx_path = './%s_model/%s'%(precision, model_name)
args = './onnx2ncnn.exe %s/model.onnx %s/model.param %s/model.bin'%(onnx_path,onnx_path,onnx_path)
print(args)
os.system(args)