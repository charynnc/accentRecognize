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
            # Add more arguments as needed
    ) -> None:
        super(CustomModel, self).__init__()
        
        # Define your encoder here
        self.encoder = TDSEResNet18(input_dim=input_dim, encoder_dim=encoder_dim, base_width=32)

        self.fc = nn.Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of model """
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
    ):
        """
        Forward propagate a `inputs` for training.

        Args:
            inputs (torch.FloatTensor): (batch, seq_length, dimension)
            input_lengths (torch.LongTensor): (batch)

        Returns:
            outputs (torch.FloatTensor): (batch, num_classes)
        """
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
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