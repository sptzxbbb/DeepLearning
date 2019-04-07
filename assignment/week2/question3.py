import torch
import numpy as np
import matplotlib.pyplot as plt

xs = torch.load('./data/xs.pt')
ys = torch.load('./data/ys.pt')
count = torch.numel(xs)
p = [[xs[i], ys[i]] for i in range(count)]
pt = torch.tensor(p)
print(pt)

x = xs.numpy()
y = ys.numpy()
z = np.polyfit(x, y, 3)
f = np.poly1d(z)

x_new = np.linspace(x[0] - 1, x[-1] + 1, 50)
y_new = f(x_new)
plt.plot(x, y, 'o', x_new, y_new)
plt.show()
