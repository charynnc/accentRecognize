import torch
import torch.nn as nn
from torch import Tensor
from typing import Any
from models.chy2 import ResNet18,SEResNet18, TDSEResNet18

class CustomModel(nn.Module):
    """
    Custom Model structure similar to Conformer.
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            use_pinyin: bool = False,
            pinyin_vocab_size: int = 0,
            pinyin_embed_dim: int = 64,
            # Add more arguments as needed
    ) -> None:
        super(CustomModel, self).__init__()
        self.use_pinyin = use_pinyin
        self.encoder_dim = encoder_dim
        
        # Define your encoder here
        self.encoder = TDSEResNet18(input_dim=input_dim, encoder_dim=encoder_dim, base_width=32)

        if self.use_pinyin:
            self.pinyin_embedding = nn.Embedding(pinyin_vocab_size, pinyin_embed_dim, padding_idx=0)
            self.pinyin_lstm = nn.LSTM(pinyin_embed_dim, encoder_dim // 2, batch_first=True, bidirectional=True)
            # Combine audio and Pinyin features
            self.fc = nn.Linear(encoder_dim * 2, num_classes, bias=False)
        else:
            self.fc = nn.Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of model """
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        pinyin_inputs: Tensor = None,
        pinyin_lengths: Tensor = None
    ):
        """
        Forward propagate a `inputs` for training.

        Args:
            inputs (torch.FloatTensor): (batch, seq_length, dimension)
            input_lengths (torch.LongTensor): (batch)
            pinyin_inputs (torch.LongTensor): (batch, pinyin_seq_length)
            pinyin_lengths (torch.LongTensor): (batch)

        Returns:
            outputs (torch.FloatTensor): (batch, num_classes)
        """
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        
        if self.use_pinyin and pinyin_inputs is not None:
            # Handle empty batch or empty sequences
            if pinyin_inputs.size(1) == 0:
                # Create dummy input of zeros (padding)
                pinyin_inputs_padded = torch.zeros(inputs.size(0), 1, dtype=torch.long, device=inputs.device)
                lens = torch.ones(inputs.size(0), dtype=torch.long, device=inputs.device)
            else:
                pinyin_inputs_padded = pinyin_inputs
                lens = pinyin_lengths.clone()
                lens[lens == 0] = 1
            
            pinyin_embed = self.pinyin_embedding(pinyin_inputs_padded)
            # Pack padded sequence
            packed_pinyin = nn.utils.rnn.pack_padded_sequence(pinyin_embed, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.pinyin_lstm(packed_pinyin)
            # hidden: (num_layers * num_directions, batch, hidden_size)
            # Concatenate forward and backward hidden states
            pinyin_feat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
            # Combine features
            combined_feat = torch.cat((encoder_outputs, pinyin_feat), dim=1)
            outputs = self.fc(combined_feat)
        else:
            outputs = self.fc(encoder_outputs)
            
        return outputs


if __name__ == "__main__":
    # Simple test
    b, t, f = 4, 300, 80
    x = torch.randn(b, t, f)
    lengths = torch.tensor([300, 250, 200, 150])

    model = CustomModel(num_classes=5, input_dim=f, encoder_dim=256)
    model.eval()

    with torch.no_grad():
        logits = model(x, lengths)

    print("Logits shape:", logits.shape)