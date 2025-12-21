import torch
from models.custom_model import CustomModel
from conformer import Conformer
from thop import profile

def calculate_flops_params():
    # Define model parameters
    num_classes = 8
    input_dim = 80
    encoder_dim = 256
    encoder_layers = 12
    attention_heads = 4
    
    # Instantiate Full Model
    vocab_sizes = {
        'initials': 30,
        'finals': 40,
        'tones': 6
    }
    
    model_full = CustomModel(
            num_classes=num_classes,
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            use_pinyin=True,
            vocab_sizes=vocab_sizes
        )
    
    # Instantiate Audio-Only Model
    model_audio = CustomModel(
            num_classes=num_classes,
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            use_pinyin=False
        )

    # Create dummy input
    batch_size = 1
    sequence_length = 300 # Example sequence length
    inputs = torch.randn(batch_size, sequence_length, input_dim)
    input_lengths = torch.tensor([sequence_length])
    
    # Dummy Pinyin inputs
    pinyin_len = 50
    initial_inputs = torch.randint(0, vocab_sizes['initials'], (batch_size, pinyin_len))
    final_inputs = torch.randint(0, vocab_sizes['finals'], (batch_size, pinyin_len))
    tone_inputs = torch.randint(0, vocab_sizes['tones'], (batch_size, pinyin_len))
    pinyin_lengths = torch.tensor([pinyin_len])

    # Calculate FLOPs and Parameters for Full Model
    flops_full, params_full = profile(model_full, inputs=(inputs, input_lengths, initial_inputs, final_inputs, tone_inputs, pinyin_lengths), verbose=False)
    
    # Calculate FLOPs for Audio Only Model
    flops_audio, params_audio = profile(model_audio, inputs=(inputs, input_lengths), verbose=False)

    print(f"Input shape: (Batch: {batch_size}, Time: {sequence_length}, Freq: {input_dim})")
    print(f"Pinyin shape: (Batch: {batch_size}, Length: {pinyin_len})")
    
    print(f"\n--- Full Model (Audio + Pinyin) ---")
    print(f"FLOPs: {flops_full / 1e9:.6f} G")
    print(f"Parameters: {params_full / 1e6:.6f} M")
    
    print(f"\n--- Audio Branch Only ---")
    print(f"FLOPs: {flops_audio / 1e9:.6f} G")
    print(f"Parameters: {params_audio / 1e6:.6f} M")
    
    print(f"\n--- Difference (Pinyin Branch + Fusion) ---")
    print(f"FLOPs: {(flops_full - flops_audio) / 1e6:.6f} M")
    print(f"Parameters: {(params_full - params_audio) / 1e6:.6f} M")
    
    # Verify with model's own count_parameters method
    model_params = model_full.count_parameters()
    print(f"\nTotal Parameters (from model.count_parameters): {model_params / 1e6:.6f} M")

if __name__ == "__main__":
    calculate_flops_params()
