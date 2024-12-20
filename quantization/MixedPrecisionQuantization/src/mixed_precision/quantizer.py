# quantization/quantizer.py
import torch
import torch.nn.functional as F
from transformers import BertModel
from copy import deepcopy
from .utils import count_parameters, calculate_model_size, calculate_reduction_rate

def quantize_model(model_name: str = 'bert-base-uncased', save_path: str = None) -> tuple:
    """
    Quantize a pre-trained model to BFloat16.
    
    Args:
        model_name (str): Name of the pre-trained model to quantize
        save_path (str, optional): Path to save the quantized model
        
    Returns:
        tuple: (quantized_model, statistics_dict)
    """
    # Load the pre-trained model
    model = BertModel.from_pretrained(model_name)
    quantized_model = deepcopy(model)
    
    # Quantize weights to BFloat16
    for name, param in quantized_model.named_parameters():
        if param.dtype == torch.float32 and 'weight' in name:
            param.data = param.data.to(torch.bfloat16)
    
    # Calculate statistics
    original_size = calculate_model_size(model)
    quantized_size = calculate_model_size(quantized_model)
    reduction_rate, reduction_times = calculate_reduction_rate(original_size, quantized_size)
    
    stats = {
        'original_size': original_size,
        'quantized_size': quantized_size,
        'reduction_rate': reduction_rate,
        'reduction_times': reduction_times,
        'original_params': count_parameters(model),
        'quantized_params': count_parameters(quantized_model)
    }
    
    # Save the model if path is provided
    if save_path:
        quantized_model.save_pretrained(save_path)
    
    return quantized_model, stats
