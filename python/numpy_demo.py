import numpy as np

def print_line():
    print("*" * 50)

a = np.array([1, 2, 3])
print(a)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

x = [1, 2, 3]
y = [[1, 2], [3, 4], [5, 6]]

x_np = np.array(x)
y_np = np.array(y)
print(type(x_np), type(x))
print(type(y_np), type(y))


x = np.array([1, 2, 3])
y = np.array([[1, 2], [3, 4], [5, 6]])
x_list = list(x_np)
y_list = list(y_np)
print(type(x_list), type(x))
print(type(y_list), type(y))


print(dir(x_np))
print(dir(x_list))


print(type(a))
print(a.shape)
print(b.shape)

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])
print(a[0], a[1], a[2])
print(a[0:1])
print(a[0:2])
print(a[0:3])
print(a[0:0])
print(a[1:2])

print_line()

print(a[:2])
print(a[1:])
print(a[:])

print_line()

print(b[0, 0], b[0, 1], b[1, 0])
print(b[:,1])
print(b[1,:])

print_line()

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])
a[0] = 5
print(a)
b[0, 2] = 10
print(b)
b[0] = 10
print(b)


print_line()

a = np.zeros((3, 2))
print(a)

b = np.ones((1, 2))
print(b)

c = np.full((2, 3), 7)
print(c)

d = np.eye(2)
print(d)

e = np.random.random((2, 3))
print(e)

e = (10 - 3) * np.random.random((2, 3)) + 3
print(e)

mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, (2, 3))
print(s)


print_line()

a = np.array([1, 2])
print(a.dtype)

b = np.array([1.0, 2])
print(b.dtype)

c = np.array([1.0, 2], dtype = np.float64)
print(c)
print(c.dtype)

d = c.astype(np.int64)
print(d)
print(d.dtype)

print_line()
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x, y)

print_line()

z = x + y
zz = np.add(x, y)

print(z)
print(zz)

print_line()
z = x - y
zz = np.subtract(x, y)

print(z)
print(zz)

print_line()

z = x * y
zz = np.multiply(x, y)
print(z)
print(zz)

print_line()
a = np.array([[1, 0], [0, 1]])
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.matmul(a, b)
print(c)

print_line()
a = np.array([[1, 0], [0, 1]])
b = np.array([1, 2])
c = np.matmul(a, b)
print(c)

from numpy.linalg import inv
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 3], [2, 4]])

ainv = inv(a)
print(ainv)

c = np.matmul(b, ainv)
print(c)

print_line()
z = x / y
zz = np.divide(x, y)
print(z)
print(zz)

print_line()

z = np.sqrt(x)
print(z)

print_line()
x = np.array([[1, 2], [3, 4]])
print(x)

print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

print_line()
a = np.array([[0, 1], [2, 3]])
b = np.amax(a)
print(b)
c = np.amax(a, axis=0)
d = np.amax(a, axis=1)
print(d)

print_line()
x = np.array([[1, 2], [3, 4]])
print(x)
print(x.T)

v = np.array([1, 2, 3])
print(v)
print(v.T)

print_line()
v = np.array([[1, 2, 3], [4, 5, 6]])
z = np.reshape(v, (3, 2))
zz = np.reshape(v, (3, -1))

print(z)
print(zz)


print_line()
v = np.array([[1, 2, 3], [4, 5, 6]])
vv = np.reshape(v, -1)
print(vv)

vv = [1, 2 ,3 ,4 ,5 ,6]
vvv = np.reshape(vv, (2, 3))
print(vvv)

import matplotlib.pyplot as plt
mu = 0.0
sigma = 0.5

xs = np.linspace(-1.0, 1.0, 100)
ys = 1 / (sigma * np.sqrt(2 * np.pi) * np.exp(- (xs - mu)**2 / (2 * sigma**2)))
print(xs)
print(xs.shape)

print(ys)
print(ys.shape)

plt.plot(xs, ys)
plt.show()