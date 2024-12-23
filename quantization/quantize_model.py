import torch
from copy import deepcopy

def quantize_model_mixed_precision(model):

    #Create a copy of the model for quantization
    quantized_model = deepcopy(model)

    #Iterate over the quantized model's named parameters and convert the weights to BFloat16
    for name, param in quantized_model.named_parameters():
        if param.dtype == torch.float32 and 'weight' in name:
            param.data = param.data.to(torch.bfloat16)

    #Save the quantized model
    quantized_model.save_pretrained('bert-base-uncased-mixed-precision')

    return quantized_model

def quantize_model_fixed_precision(model):

    #Create a copy of the model for quantization
    quantized_model = deepcopy(model)

    # Convert activations and biases as well
    quantized_model.half()

    #Save the quantized model
    quantized_model.save_pretrained('bert-base-uncased-fixed-precision')

    return quantized_model