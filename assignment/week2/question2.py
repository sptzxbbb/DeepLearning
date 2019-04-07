import random
import numpy as np

m_size = 9
kernel_size = 3
m1 = np.array([[random.randint(1, 10) for i in range(m_size)] for j in range(m_size)])
m2 = np.array([[random.randint(1, 10) for i in range(m_size)] for j in range(m_size)])
m_sum = np.zeros((m_size - 2, m_size - 2))

kernel1 = np.array([
    [1, 0, 1],
    [-1, 1, 0],
    [0, -1, 0]
])
kernel2 = np.array([
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 1]
])


for i in range(m_size - 2):
    for j in range(m_size - 2):
        for a in range(kernel_size):
            for b in range(kernel_size):
                m_sum[i][j] += m1[i + a][j + b] * kernel1[a][b]
                m_sum[i][j] += m2[i + a][j + b] * kernel2[a][b]

print("Matrix 1:\n", m1)
print("\nMatrix 2:\n", m2)
print("\nKernel 1:\n", kernel1)
print("\nKernel 2:\n", kernel2)
print("\nSum matrix:\n", m_sum)
