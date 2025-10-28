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
