from transformers import AutoModel

from utils.print_utils import (print_model_parameter_number, print_model_size, print_model_dtype, print_reduction_rate_mb,
                               print_reduction_rate_param, print_reduction_times_mb, print_reduction_times_param)
from quantization.quantize_model import quantize_model_fixed_precision

if __name__ == "__main__":
    # Load a pre-trained model (generalized for any model, not just BERT)
    model_name = "bert-base-uncased"  # Replaceable with any model name
    model = AutoModel.from_pretrained(model_name)

    # Print the dtype of the model's parameters before quantization
    print("\nBefore quantization:")
    print_model_parameter_number(model)
    print_model_size(model)
    print_model_dtype(model)

    # Perform quantization
    quantized_model = quantize_model_fixed_precision(model)

    # Print the dtype of the model's parameters AFTER quantization
    print("\nAfter quantization:")
    print_model_parameter_number(quantized_model)
    print_model_size(quantized_model)
    print_model_dtype(quantized_model)

    print("\nQuantization report:")
    print("="*50)
    print_reduction_rate_mb(model, quantized_model) #before/after
    print_reduction_rate_param(model, quantized_model) #before/after
    print_reduction_times_mb(model, quantized_model) #before/after
    print_reduction_times_param(model, quantized_model) #before/after



