# BERT Model Quantization (Mixed-Precision-Weight-Only)

This script demonstrates the process of **mixed-precision-weight-only quantization** for reducing the memory footprint of a BERT model. It specifically focuses on converting the model's weights from `float32` to `bfloat16` precision while leaving the rest of the model (e.g., activations) unaffected. This technique is useful for reducing the size of the model without sacrificing too much performance, and it is commonly used in deep learning to accelerate inference and reduce memory usage.

## Features

- Loads a pre-trained BERT model (`bert-base-uncased` by default).
- **Mixed-precision-weight-only quantization**: Converts the weights of the model from `float32` to `bfloat16`, reducing the model's memory usage.
- Displays the **reduction rate** in model size after quantization.
- Outputs detailed statistics about the model's parameters, including:
  - Shape, data type, and unique values of the embeddings.
  - Minimum and maximum values of the embeddings.
- Saves the quantized model to a specified path.

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers library

## Usage
```bash
git clone https://github.com/pierredantas/llm-compression-toolkit
pip install -r llm-compression-toolkit/requirements.txt
```
## Quantize the default BERT model and save the quantized version
```bash
python mixed_precision_quantization
```
