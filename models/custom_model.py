import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Any
from models.chy2 import ResNet18,SEResNet18, TDSEResNet18

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C)
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return x * y

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        # mask: (B, T) - 1 for valid, 0 for padding
        attn_scores = self.attention(x).squeeze(-1)  # (B, T)
        
        if mask is not None:
            # Mask padding positions with a large negative value
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        weighted = torch.sum(x * attn_weights, dim=1)  # (B, D)
        return weighted

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

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
            vocab_sizes: dict = None,
            pinyin_embed_dim: int = 128,
            dropout: float = 0.5,
            # Add more arguments as needed
    ) -> None:
        super(CustomModel, self).__init__()
        self.use_pinyin = use_pinyin
        self.encoder_dim = encoder_dim
        self.dropout_layer = nn.Dropout(dropout)
        
        # Define your encoder here
        self.encoder = TDSEResNet18(input_dim=input_dim, encoder_dim=encoder_dim, base_width=32)

        if self.use_pinyin:
            # Decomposed Embeddings
            dim_init = pinyin_embed_dim // 4
            dim_tone = pinyin_embed_dim // 4
            dim_final = pinyin_embed_dim - dim_init - dim_tone
            
            self.initial_embed = nn.Embedding(vocab_sizes['initials'], dim_init, padding_idx=0)
            self.final_embed = nn.Embedding(vocab_sizes['finals'], dim_final, padding_idx=0)
            self.tone_embed = nn.Embedding(vocab_sizes['tones'], dim_tone, padding_idx=0)

            # Transformer Encoder replacing LSTM
            self.pos_encoder = PositionalEncoding(pinyin_embed_dim, dropout)
            encoder_layers = nn.TransformerEncoderLayer(d_model=pinyin_embed_dim, nhead=4, dim_feedforward=pinyin_embed_dim*4, dropout=dropout, batch_first=True)
            self.pinyin_transformer = nn.TransformerEncoder(encoder_layers, num_layers=2)
            
            # Project Transformer output to match encoder_dim if needed
            self.pinyin_proj = nn.Linear(pinyin_embed_dim, encoder_dim)
            self.pinyin_pool = AttentionPooling(encoder_dim)
            
            # Fusion Module
            fusion_dim = encoder_dim * 2
            self.fusion_se = SEBlock(fusion_dim, reduction=16)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.BatchNorm1d(fusion_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, num_classes)
            )
        else:
            self.fc = nn.Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of model """
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        initial_inputs: Tensor = None,
        final_inputs: Tensor = None,
        tone_inputs: Tensor = None,
        pinyin_lengths: Tensor = None
    ):
        """
        Forward propagate a `inputs` for training.

        Args:
            inputs (torch.FloatTensor): (batch, seq_length, dimension)
            input_lengths (torch.LongTensor): (batch)
            initial_inputs (torch.LongTensor): (batch, pinyin_seq_length)
            final_inputs (torch.LongTensor): (batch, pinyin_seq_length)
            tone_inputs (torch.LongTensor): (batch, pinyin_seq_length)
            pinyin_lengths (torch.LongTensor): (batch)

        Returns:
            outputs (torch.FloatTensor): (batch, num_classes)
        """
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        
        if self.use_pinyin and initial_inputs is not None:
            # Handle empty batch or empty sequences
            if initial_inputs.size(1) == 0:
                # Create dummy input of zeros (padding)
                initial_inputs = torch.zeros(inputs.size(0), 1, dtype=torch.long, device=inputs.device)
                final_inputs = torch.zeros(inputs.size(0), 1, dtype=torch.long, device=inputs.device)
                tone_inputs = torch.zeros(inputs.size(0), 1, dtype=torch.long, device=inputs.device)
                lens = torch.ones(inputs.size(0), dtype=torch.long, device=inputs.device)
            else:
                lens = pinyin_lengths.clone()
                lens[lens == 0] = 1
            
            emb_init = self.initial_embed(initial_inputs)
            emb_final = self.final_embed(final_inputs)
            emb_tone = self.tone_embed(tone_inputs)
            
            pinyin_embed = torch.cat([emb_init, emb_final, emb_tone], dim=-1)
            
            # Add Positional Encoding
            pinyin_embed = self.pos_encoder(pinyin_embed)

            # Create Padding Mask for Transformer (True where padded)
            # lens: (B)
            B, T, _ = pinyin_embed.size()
            # mask: (B, T) - True for padding positions
            src_key_padding_mask = torch.arange(T, device=inputs.device).expand(B, T) >= lens.unsqueeze(1)

            # Transformer Encoder
            transformer_out = self.pinyin_transformer(pinyin_embed, src_key_padding_mask=src_key_padding_mask)
            # transformer_out: (B, T, pinyin_embed_dim)
            
            # Project to encoder dimension
            pinyin_feat_seq = self.pinyin_proj(transformer_out) # (B, T, encoder_dim)
            
            # Attention Pooling Mask (1 for valid, 0 for padding)
            pool_mask = ~src_key_padding_mask
            
            # Attention Pooling
            pinyin_feat = self.pinyin_pool(pinyin_feat_seq, pool_mask)
            pinyin_feat = self.dropout_layer(pinyin_feat)
            
            # Combine features
            encoder_outputs = self.dropout_layer(encoder_outputs)
            combined_feat = torch.cat((encoder_outputs, pinyin_feat), dim=1)
            
            # Apply SE Block to reweight channels
            combined_feat = self.fusion_se(combined_feat)
            
            # MLP Classifier
            outputs = self.fusion_mlp(combined_feat)
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