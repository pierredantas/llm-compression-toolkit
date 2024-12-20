## Overview

LLM compression is crucial for deploying these powerful models in CPU-based environments. This toolkit focuses on compression techniques that are specifically optimized for CPU inference, making it ideal for deployments where GPU resources are unavailable or impractical. All implementations are carefully designed to leverage CPU-specific optimizations and SIMD instructions (AVX2/AVX-512) where applicable.

## Features

- Model pruning techniques
  - Structured pruning
  - Unstructured pruning
  - Magnitude-based pruning
  - Movement pruning
  
- Quantization methods
  - Post-training quantization (PTQ)
  - Quantization-aware training (QAT)
  - Dynamic quantization
  - Mixed-precision quantization

- Knowledge distillation
  - Traditional knowledge distillation
  - Progressive knowledge distillation
  - Self-distillation

- Matrix/Tensor decomposition
  - SVD-based methods
  - Tensor decomposition
  - Low-rank approximation

- Fourier-based methods
  - Fast Fourier Transform (FFT) compression
  - Frequency domain pruning
  - Spectral weight clustering
  - Fourier coefficient quantization

## Getting Started

### Prerequisites

```bash
python>=3.8
torch>=1.8.0 (CPU build)
transformers>=4.0.0
intel-openmp>=2021.0
oneDNN>=2.0
```

Note: This toolkit is optimized for CPU deployment. While it can run on systems with GPUs, all optimizations target CPU inference.

### Installation

```bash
git clone https://github.com/pierredantas/llm-compression-toolkit
pip install -r llm-compression-toolkit/requirements.txt

import sys
sys.path.append('/content/llm-compression-toolkit')
```

## Usage

Each compression technique is organized in its own directory with specific documentation and examples. Here's a quick example of using quantization:

```python
from quantization.MixedPrecisionQuantization.mixed_precision_quantization import quantize_bert_model

quantized_model = quantize_bert_model(model_name='bert-base-uncased', save_path='bert-quantized')
```

## Directory Structure

```
llm-compression-toolkit/
├── pruning/
│   ├── structured/
│   ├── unstructured/
│   └── movement/
├── quantization/
│   ├── ptq/
│   ├── qat/
│   └── dynamic/
├── distillation/
├── decomposition/
├── examples/
├── tests/
├── fourier/
│   ├── fft/
│   ├── spectral/
│   └── frequency_pruning/
└── docs/
```

## CPU Benchmarks (Under Development)

All benchmarks were performed on Intel Xeon processors with AVX-512 support. Results may vary based on CPU architecture and available instruction sets.

| Compression Method | Size Reduction | Inference Speed | Quality Loss |
|-------------------|----------------|-----------------|--------------|
| 8-bit Quantization|                |    x faster     |     perplexity|
| Pruning (30%)     |                |    x faster     |     perplexity|
| Distillation      |                |    x faster     |     perplexity|
| FFT Compression   |                |    x faster     |     perplexity|

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-compression`)
3. Commit your changes (`git commit -m 'Add amazing compression method'`)
4. Push to the branch (`git push origin feature/amazing-compression`)
5. Open a Pull Request

## License

This project is licensed under the license - see the file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{llm_compression_toolkit,
  title = {LLM Compression Toolkit},
  author = {Pierre V. Dantas},
  year = {2024},
  url = {https://github.com/pierredantas/llm-compression-toolkit}
}
```

## Acknowledgments

- Thanks to all contributors and researchers in the field of model compression
- Special thanks to the open-source community for their valuable feedback and contributions
- Inspired by various research papers and implementations in the field of LLM optimization

## Contact

- GitHub Issues: For bug reports and feature requests
- Email: pierre.dantas@gmail.com
