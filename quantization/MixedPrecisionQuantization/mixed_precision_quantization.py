def quantize_bert_model(model_name: str = 'bert-base-uncased', save_path: str = 'bert-quantized'):
    import torch
    from transformers import BertModel
    from copy import deepcopy
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    def calculate_model_size(model):
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    # Load the pre-trained BERT model
    model = BertModel.from_pretrained(model_name)

    # Create a quantized version of the model
    quantized_model = deepcopy(model)
    for name, param in quantized_model.named_parameters():
        if param.dtype == torch.float32 and 'weight' in name:
            param.data = param.data.to(torch.bfloat16)

    # Print reduction statistics
    original_size = calculate_model_size(model)
    quantized_size = calculate_model_size(quantized_model)
    reduction_rate = (1 - quantized_size / original_size) * 100

    print(f"Reduction Rate: {reduction_rate:.2f}%")
    
    # Save the quantized model
    quantized_model.save_pretrained(save_path)
    print(f"Quantized model saved to {save_path}")

    # MORE STATISTICS
# Before quantization
print("\nBEFORE QUANTIZATION")
print("-" * 50)
embedding_name = "embeddings.word_embeddings.weight"
param = dict(model.named_parameters())[embedding_name]
print(f"Shape: {param.shape}")
print(f"Dtype: {param.dtype}")
print(f"All values:\n{param}")
print(f"\nMin value: {param.min()}")
print(f"Max value: {param.max()}")
print(f"Number of unique values: {len(torch.unique(param))}")

print("\nAFTER QUANTIZATION")
print("-" * 50)
param = dict(quantized_model.named_parameters())[embedding_name]
print(f"Shape: {param.shape}")
print(f"Dtype: {param.dtype}")
print(f"All values:\n{param}")
print(f"\nMin value: {param.min()}")
print(f"Max value: {param.max()}")
print(f"Number of unique values: {len(torch.unique(param))}")

    return quantized_model
