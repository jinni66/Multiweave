import torch
import torch.nn as nn
from models.transformer import TransformerEncoder
from models.transformer import CrossScaleDecoder

class MultiScaleED3(nn.Module):
    def __init__(self, dim=2, hidden=64, heads=4, dropout=0.1, factor=10):
        super().__init__()

        self.factor = factor

        self.enc_1 = TransformerEncoder(dim, hidden, 12, heads, dropout, True)
        self.enc_k = TransformerEncoder(hidden, hidden, 12, heads, dropout, False)
        self.enc_k2 = TransformerEncoder(hidden, hidden, 12, heads, dropout, False)

        self.dec_k2 = CrossScaleDecoder(hidden, heads, dropout, hidden, dim)
        self.dec_k = CrossScaleDecoder(hidden, heads, dropout, dim, dim)
        self.dec_1 = CrossScaleDecoder(hidden, heads, dropout, dim, dim)

    def downsample_sum(self, x):
        B, T, D = x.shape
        f = self.factor
        return x.view(B, T // f, f, D).sum(dim=2)

    def upsample_repeat(self, x):
        B, T, D = x.shape
        f = self.factor
        return x.unsqueeze(2).repeat(1, 1, f, 1).view(B, T * f, D)

    def forward(self, x1):
        e1 = self.enc_1(x1)

        e_k = self.enc_k(self.downsample_sum(e1))
        e_k2 = self.enc_k2(self.downsample_sum(e_k))

        y_k2 = self.dec_k2(e_k2, e_k2)
        y_k = self.dec_k(self.upsample_repeat(y_k2), e_k)
        y1 = self.dec_1(self.upsample_repeat(y_k), e1)

        return y1, y_k, y_k2
