import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_model_size(model):
    """
    Calculate the size of a model in megabytes (MB).

    Parameters:
        model (torch.nn.Module): PyTorch model.

    Returns:
        float: Model size in MB.
    """
    total_params = sum(param.numel() * param.element_size() for param in model.parameters())
    return total_params / (1024 * 1024)

def analyze_parameter_distribution(model):
    """
    Analyze the parameter distribution of a model.

    Parameters:
        model (torch.nn.Module): PyTorch model.

    Prints:
        Distribution details such as count of parameter types.
    """
    param_types = {}
    for param in model.parameters():
        dtype = param.dtype
        param_types[dtype] = param_types.get(dtype, 0) + param.numel()

    for dtype, count in param_types.items():
        print(f"{dtype}: {count} parameters")

def calculate_compression_ratio(original_params: int, quantized_params: int) -> float:
    """
    Calculate the compression ratio between the original and quantized model.

    Parameters:
    - original_params (int): Number of parameters in the original model.
    - quantized_params (int): Number of parameters in the quantized model.

    Returns:
    - compression_ratio (float): Ratio of original to quantized parameters.
    """
    if quantized_params == 0:
        raise ValueError("Quantized parameters cannot be zero.")
    compression_ratio = original_params / quantized_params
    return compression_ratio

def calculate_reduction_rate(original_size: int, quantized_size: int):
    """
    Calculate the size reduction rate and reduction factor.

    Parameters:
    - original_size (int): Size of the original model.
    - quantized_size (int): Size of the quantized model.

    Returns:
    - reduction_rate (float): Percentage of size reduction.
    - reduction_times (float): Size reduction factor.
    """
    reduction_rate = ((original_size - quantized_size) / original_size) * 100
    reduction_times = original_size / quantized_size
    return reduction_rate, reduction_times
