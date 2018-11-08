import random
import math
import numpy as np


def sigmoid(x):
    try:
        float(x)
    except TypeError:
        for index, i in enumerate(x):
            x[index] = sigmoid(i)
        return x
    else:
        y = (1 / (1 + math.exp(-x)))
        return y


def d_sigmoid(x):
    try:
        float(x)
    except TypeError:
        for index, i in enumerate(x):
            x[index] = sigmoid(i)
        return x
    else:
        y = sigmoid(x)*(1-sigmoid(x))
        return y


# 随机产生系数矩阵
def create_w(m, n):
    w = list()
    for i in range(m):
        t = list()
        w.append(t)
        for j in range(n):
            t.append(0.001*random.randint(0, 1000))
    return np.array(w)


def test(x, wx, wh, bx, bh):
    h_ = np.matmul(x, wx) + bx
    h = sigmoid(h_)
    y_ = np.matmul(h, wh.T) + bh
    y = sigmoid(y_)

    return y


# 训练过程
def train(x, y_correct, wx, wh, bx, bh):
    alpha = 0.05
    # x，wx, wh为list；y为float
    h_ = np.matmul(x, wx) + bx
    h = sigmoid(h_)
    y_ = np.matmul(h, wh.T) + bh
    y = sigmoid(y_)

    e = y_correct - y

    delta_wh = e * alpha * d_sigmoid(y) * h
    wh = wh + delta_wh
    delta_bh = e * alpha * d_sigmoid(y)
    bh = bh - delta_bh

    E = sum(e*wh)*d_sigmoid(h)
    delta_wx = E * alpha * x.reshape([2, 1])
    wx = wx + delta_wx
    delta_bx = E * alpha
    bx = bx - delta_bx

    return wx, wh, bx, bh


if __name__ == "__main__":
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    label = [0, 1, 1, 0]

    wx = create_w(2, 3)
    wh = create_w(1, 3)
    bx = create_w(1, 3)
    bh = create_w(1, 1)

    for i in range(500):
        x = random.randint(0, 3)
        wx, wh, bx, bh = train(np.array(data[x]), label[x], wx, wh, bx, bh)

    for i in range(4):
        print("x=", data[i])
        print("y=", test(data[i], wx, wh, bx, bh))

