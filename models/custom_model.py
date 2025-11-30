import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class CustomModel(nn.Module):
    """
    Custom Model structure similar to Conformer.
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            # Add more arguments as needed
    ) -> None:
        super(CustomModel, self).__init__()
        
        # TODO: Define your encoder here
        # self.encoder = ...
        self.encoder = None 

        self.fc = nn.Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of model """
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Forward propagate a `inputs` for training.

        Args:
            inputs (torch.FloatTensor): (batch, seq_length, dimension)
            input_lengths (torch.LongTensor): (batch)

        Returns:
            outputs (torch.FloatTensor): (batch, num_classes)
        """
        # TODO: Implement encoder forward pass
        # encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        
        # Placeholder logic (assuming encoder_outputs is available):
        # outputs = encoder_outputs.mean(dim=1)
        # outputs = self.fc(outputs)
        # return outputs
        
        raise NotImplementedError("Encoder is not implemented. Please implement the encoder in __init__ and forward.")
