from transformers import AutoModel

# Utility functions for metrics and testing
from utils.metrics import calculate_compression_ratio, calculate_reduction_rate, count_parameters, calculate_model_size, analyze_parameter_distribution
from utils.print_utils import print_before_quantization, print_after_quantization, print_statistics
from quantization.quantize_model import quantize_model

class MixedPrecisionQuantizer:
    """Class for mixed-precision quantization of models."""

    def __init__(self, model):
        self.model = model
        self.original_size = calculate_model_size(model)

    def quantize(self):
        """Apply mixed-precision quantization to the model."""
        self.model = quantize_model(self.model)
        self.quantized_size = calculate_model_size(self.model)

    def report_statistics(self):
        """Report statistics pre- and post-quantization."""
        reduction_rate, reduction_times = calculate_reduction_rate(self.original_size, self.quantized_size)
        print_statistics(self.original_size, self.quantized_size, (reduction_rate, reduction_times))

if __name__ == "__main__":
    # Load a pre-trained model (generalized for any model, not just BERT)
    model_name = "bert-base-uncased"  # Replaceable with any model name
    model = AutoModel.from_pretrained(model_name)

    # Print the dtype of the model's parameters before quantization
    print_before_quantization(model)

    # Initialize the quantizer
    quantizer = MixedPrecisionQuantizer(model)

    # Perform quantization
    quantizer.quantize()

    # Print the dtype of the quantized model's parameters after quantization
    print_after_quantization(quantizer.model)

    # Report statistics
    quantizer.report_statistics()

    # Save the quantized model
    quantizer.model.save_pretrained('bert-base-uncased-fp16')
