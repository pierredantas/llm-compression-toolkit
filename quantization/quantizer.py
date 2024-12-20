import torch
from transformers import PreTrainedModel
from typing import Tuple, Optional, Dict
from .utils import calculate_model_size, count_parameters, calculate_reduction_rate

class ModelQuantizer:
    """A class for quantizing PyTorch models to reduced precision formats."""
    
    def __init__(self, model: PreTrainedModel):
        """
        Initialize the quantizer with a model.
        
        Args:
            model (PreTrainedModel): The model to be quantized
        """
        self.original_model = model
        self.quantized_model = None
        self.original_size = calculate_model_size(model)
        self.quantized_size = None
        
    def quantize(self, dtype: torch.dtype = torch.bfloat16, 
                 weights_only: bool = True) -> PreTrainedModel:
        """
        Quantize the model to the specified dtype.
        
        Args:
            dtype (torch.dtype): Target dtype for quantization
            weights_only (bool): If True, only quantize the weights, not biases and activations
            
        Returns:
            PreTrainedModel: The quantized model
        """
        import copy
        self.quantized_model = copy.deepcopy(self.original_model)
        
        if not weights_only:
            self.quantized_model = self.quantized_model.to(dtype)
        else:
            for name, param in self.quantized_model.named_parameters():
                if 'weight' in name and param.dtype == torch.float32:
                    param.data = param.data.to(dtype)
                    
        self.quantized_size = calculate_model_size(self.quantized_model)
        return self.quantized_model
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the quantization process.
        
        Returns:
            Dict: Dictionary containing various statistics
        """
        if self.quantized_model is None:
            raise ValueError("Model has not been quantized yet. Call quantize() first.")
            
        reduction_rate, reduction_times = calculate_reduction_rate(
            self.original_size, self.quantized_size
        )
        
        return {
            "original_parameters": count_parameters(self.original_model),
            "quantized_parameters": count_parameters(self.quantized_model),
            "original_size_bytes": self.original_size,
            "quantized_size_bytes": self.quantized_size,
            "reduction_rate_percent": reduction_rate,
            "reduction_times": reduction_times
        }
        
    def save_model(self, path: str):
        """
        Save the quantized model to the specified path.
        
        Args:
            path (str): Path where to save the model
        """
        if self.quantized_model is None:
            raise ValueError("Model has not been quantized yet. Call quantize() first.")
        
        self.quantized_model.save_pretrained(path)
