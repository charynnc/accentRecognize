import torch
from conformer import Conformer

batch_size = 4
sequence_length = 100
dim = 80
num_classes = 8

model = Conformer(num_classes=num_classes, 
                  input_dim=dim, 
                  encoder_dim=32, 
                  num_encoder_layers=3)

inputs = torch.rand(batch_size, sequence_length, dim)
input_lengths = torch.LongTensor([100, 90, 80, 70])

outputs = model(inputs, input_lengths)

print(f"Output shape: {outputs.shape}")
print(f"Expected shape: ({batch_size}, {num_classes})")

assert outputs.shape == (batch_size, num_classes)
print("Test passed!")
