from utils.metrics import (count_parameters, calculate_model_size, calculate_reduction_rate_mb, calculate_reduction_rate_param,
                           calculate_reduction_times_mb, calculate_reduction_times_param)

def print_model_dtype(model):
    print("")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

def print_model_parameter_number(model):
    print(f"\n--> Parameter number: {count_parameters(model):,.0f}")

def print_model_size(model):
    print(f"\n--> Model size: {calculate_model_size(model):,.0f} bytes")

def print_reduction_rate_mb(before_model, after_model):
    print(f"\n--> Reduction size rate (@MB): {calculate_reduction_rate_mb(before_model, after_model):.4f}%")

def print_reduction_rate_param(before_model, after_model):
    print(f"--> Reduction parameters rate (@param): {calculate_reduction_rate_param(before_model, after_model):.4f}%")

def print_reduction_times_mb(before_model, after_model):
    print(f"\n--> Reduction times (@MB): {calculate_reduction_times_mb(before_model, after_model):.4f}x")

def print_reduction_times_param(before_model, after_model):
    print(f"--> Reduction times (@param): {calculate_reduction_times_param(before_model, after_model):.4f}x")





