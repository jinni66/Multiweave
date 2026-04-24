import torch
import torch.nn as nn

# =====================================================
# Transformer Encoder
# =====================================================
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, use_input_proj=True):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size) if use_input_proj else nn.Identity()
        self.pos_embedding = nn.Embedding(2000000, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.encoder(x + self.pos_embedding(pos))


class CrossScaleDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, input_size, output_size):
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(2000000, hidden_size)

        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, enc_feat):
        x = self.embedding(x)
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        h, _ = self.self_attn(x, x, x)
        x = self.norm1(x + h)

        h, _ = self.cross_attn(x, enc_feat, enc_feat)
        x = self.norm2(x + h)

        h = self.ffn(x)
        x = self.norm3(x + h)

        return self.out(x)