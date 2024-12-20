# File: mixed_precision_quantization.py

import torch
import torch.nn.functional as F
from transformers import BertModel
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

class MixedPrecisionQuantizer:
    """A class for mixed precision quantization of transformer models."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize the quantizer with a model name.
        
        Args:
            model_name (str): Name of the pre-trained model to load
        """
        self.original_model = BertModel.from_pretrained(model_name)
        self.quantized_model = None
        self.model_name = model_name
        
    @staticmethod
    def count_parameters(model) -> int:
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def calculate_model_size(model) -> int:
        """Calculate the model size in bytes."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return param_size + buffer_size
    
    @staticmethod
    def calculate_reduction_rate(original_size: int, quantized_size: int) -> Tuple[float, float]:
        """Calculate reduction rate and times."""
        reduction_rate = (1 - quantized_size / original_size) * 100
        reduction_times = original_size / quantized_size
        return reduction_rate, reduction_times
    
    def quantize(self, dtype: torch.dtype = torch.bfloat16) -> None:
        """
        Quantize the model to the specified dtype.
        
        Args:
            dtype (torch.dtype): Target dtype for quantization
        """
        self.quantized_model = deepcopy(self.original_model)
        for name, param in self.quantized_model.named_parameters():
            if param.dtype == torch.float32 and 'weight' in name:
                param.data = param.data.to(dtype)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the quantization."""
        if self.quantized_model is None:
            raise ValueError("Model must be quantized first. Call quantize() method.")
            
        original_size = self.calculate_model_size(self.original_model)
        quantized_size = self.calculate_model_size(self.quantized_model)
        reduction_rate, reduction_times = self.calculate_reduction_rate(original_size, quantized_size)
        
        stats = {
            'original_params': self.count_parameters(self.original_model),
            'quantized_params': self.count_parameters(self.quantized_model),
            'original_size': original_size,
            'quantized_size': quantized_size,
            'reduction_rate': reduction_rate,
            'reduction_times': reduction_times,
            'dtype_info': {
                'before': {name: param.dtype for name, param in self.original_model.named_parameters()},
                'after': {name: param.dtype for name, param in self.quantized_model.named_parameters()}
            }
        }
        return stats
    
    def save_model(self, output_path: str) -> None:
        """
        Save the quantized model.
        
        Args:
            output_path (str): Path to save the quantized model
        """
        if self.quantized_model is None:
            raise ValueError("Model must be quantized first. Call quantize() method.")
        self.quantized_model.save_pretrained(output_path)

def quantize_model(model_name: str = 'bert-base-uncased', 
                  dtype: torch.dtype = torch.bfloat16,
                  output_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function to quantize a model in one line.
    
    Args:
        model_name (str): Name of the pre-trained model
        dtype (torch.dtype): Target dtype for quantization
        output_path (str, optional): Path to save the quantized model
        
    Returns:
        tuple: (quantized_model, statistics)
    """
    quantizer = MixedPrecisionQuantizer(model_name)
    quantizer.quantize(dtype)
    stats = quantizer.get_statistics()
    
    if output_path:
        quantizer.save_model(output_path)
        
    return quantizer.quantized_model, stats
