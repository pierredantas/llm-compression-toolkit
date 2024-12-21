import torch
from utils.metrics import calculate_compression_ratio, calculate_reduction_rate, count_parameters, calculate_model_size, analyze_parameter_distribution

def print_before_quantization(model):
    """
    Print information about the model before quantization.
    """
    print("\nBefore quantization:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    print(f"\nParameter number: {count_parameters(model):,.0f}")
    print(f"Model size: {calculate_model_size(model):,.0f} bytes")

def print_after_quantization(model):
    """
    Print information about the model after quantization.
    """
    print("\nAfter quantization:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    print(f"\nParameter number: {count_parameters(model):,.0f}")
    print(f"Model size: {calculate_model_size(model):,.0f} bytes")

def print_statistics(original_size, quantized_size, reduction_rate):
    """
    Print model size statistics before and after quantization.
    """
    print("\nReduction Statistics")
    print("-" * 50)
    reduction_rate, reduction_times = reduction_rate
    print(f"Original model size: {original_size:,.0f} bytes")
    print(f"Quantized model size: {quantized_size:,.0f} bytes")
    print(f"Reduction size rate: {reduction_rate:.4f}%")
    print(f"Reduction size times: {reduction_times:.4f}x")
