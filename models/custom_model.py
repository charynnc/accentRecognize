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

class DecomposedPhoneticEncoder(nn.Module):
    """
    Encodes decomposed pinyin sequences (Initials, Finals, Tones) into a fixed-size vector.
    """
    def __init__(self, vocab_sizes, embed_dim, output_dim, dropout=0.1):
        super(DecomposedPhoneticEncoder, self).__init__()
        
        dim_init = embed_dim // 4
        dim_tone = embed_dim // 4
        dim_final = embed_dim - dim_init - dim_tone
        
        self.initial_embed = nn.Embedding(vocab_sizes['initials'], dim_init, padding_idx=0)
        self.final_embed = nn.Embedding(vocab_sizes['finals'], dim_final, padding_idx=0)
        self.tone_embed = nn.Embedding(vocab_sizes['tones'], dim_tone, padding_idx=0)

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=2)
        
        self.proj = nn.Linear(embed_dim, output_dim)
        self.pool = AttentionPooling(output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, initial_inputs, final_inputs, tone_inputs, lengths):
        # Embeddings
        emb_init = self.initial_embed(initial_inputs)
        emb_final = self.final_embed(final_inputs)
        emb_tone = self.tone_embed(tone_inputs)
        
        x = torch.cat([emb_init, emb_final, emb_tone], dim=-1)
        
        # Positional Encoding
        x = self.pos_encoder(x)

        # Padding Mask
        B, T, _ = x.size()
        src_key_padding_mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Projection
        x = self.proj(x)
        
        # Pooling
        pool_mask = ~src_key_padding_mask
        x = self.pool(x, pool_mask)
        
        # Norm & Dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x

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
            self.phonetic_encoder = DecomposedPhoneticEncoder(
                vocab_sizes=vocab_sizes,
                embed_dim=pinyin_embed_dim,
                output_dim=encoder_dim,
                dropout=dropout
            )
            
            # Normalization for Fusion
            self.audio_norm = nn.LayerNorm(encoder_dim)
            # text_norm is now inside DecomposedPhoneticEncoder

            # Fusion Module
            # Concat(Audio, Text, Audio*Text) -> 3 * encoder_dim
            fusion_dim = encoder_dim * 3
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
            
            # Get Linguistic Features
            pinyin_feat = self.phonetic_encoder(initial_inputs, final_inputs, tone_inputs, lens)
            
            # Normalize Audio features before fusion
            encoder_outputs = self.audio_norm(encoder_outputs)
            # pinyin_feat is already normalized by DecomposedPhoneticEncoder

            # Bilinear Fusion: Concat(A, T, A*T)
            interaction = encoder_outputs * pinyin_feat
            combined_feat = torch.cat((encoder_outputs, pinyin_feat, interaction), dim=1)
            
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