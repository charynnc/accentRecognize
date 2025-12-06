import torch
from models.custom_model import CustomModel
from conformer import Conformer
from thop import profile

def calculate_flops_params():
    # Define model parameters
    num_classes = 8
    input_dim = 80
    encoder_dim = 512
    encoder_layers = 6
    attention_heads = 4
    
    # Instantiate the model
    # model = CustomModel(num_classes=num_classes, 
    #                     input_dim=input_dim, 
    #                     encoder_dim=encoder_dim)
    model = Conformer(num_classes=num_classes,
                      input_dim=input_dim,
                      encoder_dim=encoder_dim,
                      num_encoder_layers=encoder_layers,
                      num_attention_heads=attention_heads)
    
    # Create dummy input
    batch_size = 1
    sequence_length = 200 # Example sequence length
    inputs = torch.randn(batch_size, sequence_length, input_dim)
    input_lengths = torch.tensor([sequence_length])
    
    # Calculate FLOPs and Parameters
    # thop.profile returns (flops, params)
    # We need to pass inputs as a tuple
    flops, params = profile(model, inputs=(inputs, input_lengths), verbose=False)
    
    print(f"Model: CustomModel (ResNet50 Encoder)")
    print(f"Input shape: (Batch: {batch_size}, Time: {sequence_length}, Freq: {input_dim})")
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.2f} M")
    
    # Verify with model's own count_parameters method
    model_params = model.count_parameters()
    print(f"Parameters (from model.count_parameters): {model_params / 1e6:.2f} M")

if __name__ == "__main__":
    calculate_flops_params()
