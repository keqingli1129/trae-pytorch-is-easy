import torch
import torch.nn as nn

B, T, C = 4, 8, 3
vocab_size = 10
block_size = 8

token_embedding = nn.Embedding(vocab_size, C)
positional = nn.Embedding(block_size, C)

idx = torch.randint(0, vocab_size, (B, T))
tok_emb = token_embedding(idx)
pos = torch.arange(0, T, dtype=torch.long)
pos_emb = positional(pos)


x = tok_emb + pos_emb
print("Shape of idx:", idx.shape)
print("Shape of pos:", pos.shape)
print("Shape of tok_emb:", tok_emb.shape)
print("Shape of pos_emb:", pos_emb.shape)
print("Shape of x:", x.shape)