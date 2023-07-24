import numpy as np


def softmax(x):
    x_exp = np.exp(x)
    aa = x_exp / np.sum(x_exp)  # softmax函数实现
    return aa


ip = np.array([1, 2, 3, 4, 5])  # 输入任意N维度向量
op = softmax(ip)  # 输出N维度向量
print(op)
print(ip.shape, op.shape)  # 输入输出形状都为(n,)
