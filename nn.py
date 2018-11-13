import random
import math
import numpy as np
import matplotlib.pyplot as plt


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
def train(x, y_correct, wx, wh):
    alpha = 0.5

    # 正向
    x = np.vstack((x, np.array([1])))
    h_ = np.matmul(wx, x)
    h = sigmoid(h_)
    h = np.vstack((h, np.array([1])))
    y_ = np.matmul(wh, h)
    y = sigmoid(y_)

    e = (y_correct - y) * y * (1-y)
    delta_wh = np.dot(-abs(alpha * e), wh)
    wh = wh + delta_wh
    delta_wx =


if __name__ == "__main__":
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    label = [0, 1, 1, 0]

    wx = create_w(2, 3)
    wh = create_w(1, 3)

    for i in range(100):
        x = random.randint(0, 3)
        wx, wh= train(np.array(data[x]).reshape([2, 1]), label[x], wx, wh)

    for i in range(4):
        print("x=", data[i])
        print("y=", test(np.array(data[i]).reshape([2, 1]), wx, wh))
    print("w1=", wx)
    print("w2=", wh)