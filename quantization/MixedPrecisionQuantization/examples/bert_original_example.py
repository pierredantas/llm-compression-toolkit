import torch
from transformers import BertTokenizer, BertForMaskedLM

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses most warnings and info logs

# Initialize tokenizer and model
def load_model_and_tokenizer():
    """
    Loads the BERT tokenizer and the pre-trained BERT model for Masked Language Modeling (MLM).
    
    Returns:
        tokenizer (BertTokenizer): The BERT tokenizer.
        model (BertForMaskedLM): The pre-trained BERT model for MLM.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    return tokenizer, model

# Helper function for MLM predictions
def predict_masked_word(text, tokenizer, model):
    """
    Given a text with a masked token, predict the most likely words that can fill the mask.
    
    Args:
        text (str): The input text with [MASK] token.
        tokenizer (BertTokenizer): The BERT tokenizer.
        model (BertForMaskedLM): The BERT model for Masked Language Modeling.
    
    Returns:
        list: Top 3 predictions for the masked token with their probabilities.
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

        # Get predictions for masked token
        mask_token_logits = predictions[0, mask_token_index, :]

        # Get top tokens (excluding special tokens)
        top_logits, top_indices = torch.topk(mask_token_logits, 100)  # Get more candidates first
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

# Test examples
mlm_examples = [
    "The predictions are completely [MASK].",  # False
    "She plays the [MASK] beautifully in the orchestra.",  # violin, piano
    "The cat caught a [MASK] in the garden.",  # mouse, bird
    "I need to [MASK] my teeth before going to bed.",  # brush
    "The [MASK] building in the world is in Dubai.",  # tallest
    "The [MASK] is barking at the mailman.",  # dog
    "I like to drink [MASK] with my breakfast.",  # coffee, tea
    "Paris is the capital of [MASK].",  # France
    "The Earth orbits around the [MASK].",  # sun
    "Water [MASK] at 100 degrees Celsius.",  # boils
    "The [MASK] went extinct millions of years ago.",  # dinosaurs
    "Please [MASK] the door when you leave.",  # close, lock
    "The [MASK] Ocean is the largest on Earth.",  # Pacific
    "Shakespeare wrote many famous [MASK].",  # plays
    "Mount Everest is the [MASK] mountain in the world."  # highest
]

def main():
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    print("=== Masked Language Modeling Examples ===")
    for i, example in enumerate(mlm_examples, 1):
        print(f"\nMLM Example {i}:")
        print(f"Input: {example}")
        predictions = predict_masked_word(example, tokenizer, model)
        print("Top 3 predictions:", predictions)

if __name__ == "__main__":
    main()
