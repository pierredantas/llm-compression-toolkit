# BERT Model Quantization (Fixed-Precision)

This script demonstrates the process of **fixed-precision quantization** for reducing the memory footprint of a BERT model. It specifically focuses on converting the model's parameters from `float32` to `float16` precision. This technique is useful for reducing the size of the model without sacrificing too much performance, and it is commonly used in deep learning to accelerate inference and reduce memory usage.

## Features

- Loads a pre-trained BERT model (`bert-base-uncased` by default).
- **Fixed-precision quantization**: Converts the model from `float32` to `bfloat16`, reducing the model's memory usage.
- Displays the **reduction rate** in model size after quantization.
- Outputs detailed statistics about the model's parameters, including:
  - Shape, data type, and unique values of the embeddings.
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
python fixed_precision_quantization
```
