import torch
from transformers import BertTokenizer, BertForMaskedLM

# Create MLM model from base model
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Replace the bert encoder with your quantized model
mlm_model.bert = quantized_model

# Convert the entire MLM model to BFloat16, including all components
def convert_model_to_bfloat16(model):
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data = module.weight.data.to(torch.bfloat16)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data.to(torch.bfloat16)
    return model

# Convert the entire MLM model to BFloat16
mlm_model = convert_model_to_bfloat16(mlm_model)

# Modified helper function for BFloat16 predictions
def predict_masked_word(text, model):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")

    # Only convert attention_mask to BFloat16, keep input_ids as Long
    if 'attention_mask' in inputs:
        inputs['attention_mask'] = inputs['attention_mask'].to(torch.bfloat16)

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()

    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        # Convert predictions back to float32 for post-processing
        predictions = outputs.logits.float()

    # Get predictions for masked token
    mask_token_logits = predictions[0, mask_token_index, :]

    # Get top tokens (excluding special tokens)
    top_logits, top_indices = torch.topk(mask_token_logits, 100)
    probs = torch.softmax(top_logits, dim=1)

    # Filter out special tokens and get top 3
    valid_predictions = []
    for i, token_id in enumerate(top_indices[0]):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        if not token.startswith('[') and not token.startswith('##'):
            valid_predictions.append((token, probs[0][i].item()))
        if len(valid_predictions) == 3:
            break

    return valid_predictions

# Set model to evaluation mode
mlm_model.eval()

# Test examples with error handling
print("=== Masked Language Modeling Examples (BFloat16 Model) ===")
for i, example in enumerate(mlm_examples, 1):
    try:
        print(f"\nMLM Example {i}:")
        print(f"Input: {example}")
        predictions = predict_masked_word(example, mlm_model)
        print("Top 3 predictions:", predictions)
    except RuntimeError as e:
        print(f"Error in example {i}: {str(e)}")
