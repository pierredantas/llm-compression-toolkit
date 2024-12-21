from enum import Enum

class CompressionType(Enum):
    """
    Enum for different types of compression methods.
    """
    QUANTIZATION = "Quantization"
    PRUNING = "Pruning"
    DISTILLATION = "Distillation"
