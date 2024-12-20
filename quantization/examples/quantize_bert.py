from transformers import BertModel
from quantization import ModelQuantizer

def main():
    # Load the pre-trained BERT model
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Initialize quantizer
    quantizer = ModelQuantizer(model)
    
    # Quantize the model
    quantized_model = quantizer.quantize(dtype=torch.bfloat16, weights_only=True)
    
    # Get statistics
    stats = quantizer.get_statistics()
    
    # Print results
    print("\nQuantization Statistics:")
    print("-" * 50)
    print(f"Original parameters: {stats['original_parameters']:,}")
    print(f"Quantized parameters: {stats['quantized_parameters']:,}")
    print(f"Original size: {stats['original_size_bytes']:,} bytes")
    print(f"Quantized size: {stats['quantized_size_bytes']:,} bytes")
    print(f"Reduction rate: {stats['reduction_rate_percent']:.2f}%")
    print(f"Reduction factor: {stats['reduction_times']:.2f}x")
    
    # Save the quantized model
    quantizer.save_model('bert-base-uncased-quantized')

if __name__ == "__main__":
    main()
