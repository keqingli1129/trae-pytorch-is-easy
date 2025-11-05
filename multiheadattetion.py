import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        return self.dropout(x)