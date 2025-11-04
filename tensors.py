import torch

data = [[1,2,3], [4,5,6]]

my_tensor = torch.tensor(data)
print(my_tensor)

shape = (2, 3)

ones = torch.ones(shape)
print(ones)
zeros = torch.zeros(shape)
print(zeros)
random = torch.randn(shape)
print(random)

template = torch.randint_like(random, low=0, high=10, dtype=torch.int32)  # 0-9
print(template)

print(f"Shape of my_tensor: {my_tensor.shape}")
print(f"Data type of my_tensor: {my_tensor.dtype}")
print(f"Device of my_tensor: {my_tensor.device}")