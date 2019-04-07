import torch

x = torch.empty(4, 4)
print(x)


x = torch.randn(4, 4)
print(x)

x = torch.zeros(4, 4, dtype=torch.long)

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int)
print(x)

x = torch.eye(3)
print(x)

x = x.new_zeros(5, 3, dtype=torch.double)
print(x)

x1 = torch.randn_like(x, dtype=torch.float)
print(x1)

x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = x + y
print(z)

z = torch.add(x, y)
print(z)


result = torch.empty(3, 3)
torch.add(x, y, out=result)

y.add_(x)
print(y)

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = torch.Tensor([1, 1, 1])

y = torch.empty_like(x)

print(x.size())

for i in range(x.size(0)):
    y[i, :] = x[i, :] + v

print(y)


x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = torch.Tensor([[1, 1, 1]])
v1 = torch.Tensor([1, 1, 1])
y = x + v
y1 = x + v1
print(y)
print(y1)

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = torch.Tensor([[1], [2], [3], [4]])
y = x + v


x = torch.Tensor([2, 3, 3])
y = torch.Tensor([2, 3, 4])

print(torch.dot(x, y))


x = torch.Tensor([[1, 2, 3], [1, 2, 3]])
y = torch.Tensor([[2, 2, 2], [2, 2, 2]])
print(x * y)
print(x / y)

x = torch.Tensor([[2, 3], [1, 1]])
y = torch.Tensor([[2, 2, 1], [1, 1, 1]])
print(torch.mm(x, y))

x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
print(x.sum())
print(x.sum(dim=0))
print(x.sum(dim=1))


x = torch.randn(3, 4)
print(x)
print(torch.transpose(x, 1, 0))

x = torch.randn(3, 4)
print(x)
print(x[:, 1])

x = torch.randn(4, 4, dtype=torch.float)
y = x.view(16)
print(y.size())

z = x.view(-1, 8)
t = x.view(8, -1)
print(z.size(), t.size())

x = torch.randn(4, 4, dtype=torch.float)
y = x.reshape(16)
print(y.size())


x = torch.randn(1)
print(x)

y = x.item()
print(y)

t = torch.Tensor([[1, 2]])
x = torch.squeeze(t)
print(x)

y = torch.unsqueeze(x, dim=0)
print(x)

x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())

y = torch.squeeze(x)
print(y.size())

y = torch.unsqueeze(y, dim=0)
print(z.size())


import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
print(b)

a = torch.ones(5)
b = a.numpy()
print(b)

import torchvision

device1 = torch.device('cuda:0')
device2 = torch.device('cpu')

x = torch.randn(5, 3)
print(x.device)

print("*" * 50)

if torch.cuda.is_available():
    x = x.to(device1)
    print(x.device)

x = x.to(device2)
print(x.device)

t = torch.tensor([1, 2], device=device1)
print(t.device)


model = torchvision.models.vgg19()

device = torch.device('cuda:0')
model1 = torchvision.models.vgg11().to(device1)