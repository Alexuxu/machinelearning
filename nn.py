import random
import math
import numpy as np


def add(x1, x2):
    y = list()
    for i in range(len(x1)):
        t = x1[i]*x2[i]
        y.append(t)
    return np.array(y)


def sigmoid(x):
    try:
        x = float(x)
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
    h_ = np.matmul(wx.T, x) + bx
    h = sigmoid(h_)
    y_ = np.matmul(wh, h) + bh
    y = sigmoid(y_)

    return y


# 训练过程
def train(x, y_correct, wx, wh, bx, bh):
    alpha = 0.5

    # 正向
    h_ = np.matmul(wx.T, x) + bx
    h = sigmoid(h_)
    y_ = np.matmul(wh, h) + bh
    y = sigmoid(y_)

    loss = (y - y_correct)
    print(loss)

    # 反向
    e = y*(1-y)*(y-y_correct)
    delta_wh = alpha*e*wh
    wh = wh+delta_wh
    delta_bh = alpha*e
    bh = bh+delta_bh

    E = add(add(h, (1-h)), wh.reshape([2, 1]))
    delta_wx = alpha*e*np.matmul(x, E.T)
    wx = wx+delta_wx
    delta_bx = alpha*e*E
    bx = bx+delta_bx

    return wx, wh, bx, bh


if __name__ == "__main__":
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    label = [0, 1, 1, 0]

    wx = create_w(2, 2)
    wh = create_w(1, 2)
    bx = create_w(2, 1)
    bh = create_w(1, 1)

    for i in range(10):
        x = random.randint(0, 3)
        wx, wh, bx, bh = train(np.array(data[x]).reshape([2, 1]), label[x], wx, wh, bx, bh)

    for i in range(4):
        print("x=", data[i])
        print("y=", test(np.array(data[i]).reshape([2, 1]), wx, wh, bx, bh))
