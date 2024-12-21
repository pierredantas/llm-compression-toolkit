import torch

def quantize_model(model):
    """
    Apply BFloat16 mixed-precision quantization to the given model.

    Args:
        model (torch.nn.Module): The model to quantize.

    Returns:
        torch.nn.Module: The quantized model.
    """
    model = model.to(torch.bfloat16)
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    return model
