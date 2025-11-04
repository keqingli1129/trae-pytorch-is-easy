import torch
import torch.nn as nn
import math
from dataclasses import dataclass

class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = nn.Dropout(config.dropout)
        self.qkv_proj = nn.Linear(self.n_embed, self.n_embed * 3, bias=False)
        # self.out_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)   
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # each is (B, T, C)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, T, T)
        att = att.masked_fill(self.get_causal_mask(T, x.device), float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, T, C)
        # y = self.out_proj(y)
        return y

    def get_causal_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask
