import torch

t = torch.tensor([range(10, 50)], dtype=torch.float64)
print(t)
print(torch.max(t))
print(torch.min(t))

