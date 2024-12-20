# quantization/utils.py
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size

def calculate_reduction_rate(original_size, quantized_size):
    reduction_rate = (1 - quantized_size / original_size) * 100
    reduction_times = original_size / quantized_size
    return reduction_rate, reduction_times
