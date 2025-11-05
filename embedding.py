import torch
import torch.nn as nn

if __name__ == "__main__":
    vocab_size = 10
    n_embd = 3

    token_embedding = nn.Embedding(vocab_size, n_embd)
    print("shape of our coordinate book:", token_embedding.weight.shape)
    print("Content of the book:")
    print(token_embedding.weight)

