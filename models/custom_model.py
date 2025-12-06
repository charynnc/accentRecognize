import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from models.chy import ResNet50

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
        
        # Define your encoder here
        self.encoder = ResNet50(input_dim=input_dim, encoder_dim=encoder_dim)

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
        # Implement encoder forward pass
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        
        # encoder_outputs is (batch, encoder_dim)
        outputs = self.fc(encoder_outputs)
        return outputs
