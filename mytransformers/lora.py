import torch
import torch.nn as nn   
import math
import torch.nn.functional as F
import torch.linalg as linalg

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.randn(rank, out_features))
        self.scaling = self.alpha / self.rank

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return x@self.A@self.B*self.scaling

class LinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, freeze_weights=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if freeze_weights:
            self.linear.requires_grad_(False)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.dropout(self.linear(x) + self.lora(x))

def test_lora():
    in_features = 100
    out_features = 10
    rank = 4
    alpha = 1.0
    freeze_weights = True
    lora = LinearWithLoRA(in_features, out_features, rank, alpha, freeze_weights)
    x = torch.randn(1, in_features)
    y = lora(x)
    assert y.shape == (1, out_features)

    original_params = sum(p.requires_grad for p in lora.linear.parameters())
    assert original_params == 0, "Original linear layer parameters not frozen"
    lora_params = sum(p.requires_grad for p in lora.lora.parameters())
    assert lora_params == 2, "LoRA parameters not trainable"

    lora_contribution = lora.lora(x)
    if len(lora_contribution.shape) > 2:
        lora_contribution = lora_contribution.reshape(-1, lora_contribution.shape[-1])
    rank_effective = linalg.matrix_rank(lora_contribution)
    assert rank_effective <= rank, "Effective rank of LoRA contribution does not match specified rank"


if __name__ == "__main__":
    print(test_lora())
    input_size = 128
    output_size = 64
    batch_size = 32

    x = torch.randn(batch_size, input_szie)

    lora_model = LinearWithLoRA(input_szie, output_size, rank=8, alpha=16)
    y = lora_model(x)
    assert y.shape == (batch_size, output_size)

    output = lora_model(x)

    print(output.shape)

