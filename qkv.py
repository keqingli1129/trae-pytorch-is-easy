import torch
import torch.nn as nn
import math
from dataclasses import dataclass

B, T, C = 1, 4, 2
x = torch.tensor([[1,2], [3,4], [5,6], [7,8]]).float()

torch.manual_seed(42)

q_proj = nn.Linear(C, C, bias=False)
q_proj.weight.data = torch.randn(C, C)
q = q_proj(x)

k_proj = nn.Linear(C, C, bias=False)
k_proj.weight.data = torch.randn(C, C)
k = k_proj(x)

v_proj = nn.Linear(C, C, bias=False)
v_proj.weight.data = torch.randn(C, C)
v = v_proj(x)

print(x)
print(x.transpose(0, 1))


