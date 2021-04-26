# -*- coding:utf-8-*-
import torch
import onnx
from onnx_coreml import convert

input_shape = (3, 16, 16)

a = torch.zeros(1, *input_shape)
print(a)



