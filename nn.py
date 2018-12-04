import random
import math
import numpy as np
import matplotlib.pyplot as plt

plotx = []
ploty = []

def sgn(x):
    if x>=0.5:
        return 1
    else:
        return 0


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
            t.append(0.002 * random.randint(0, 1000) - 1)
    return np.array(w)


def test(x, wx, wh):
    x = np.vstack((x, np.array([1])))
    h_ = np.matmul(wx, x)
    h = sigmoid(h_)
    h = np.vstack((h, np.array([1])))
    y_ = np.matmul(wh, h)
    y = sigmoid(y_)

    return sgn(y)


# 训练过程
def train(x, y_correct, wx, wh, i):
    alpha = 0.5
    global plotx, ploty

    # 正向
    x = np.vstack((x, np.array([1])))
    h_ = np.matmul(wx, x)
    h = sigmoid(h_)
    h = np.vstack((h, np.array([1])))
    y_ = np.matmul(wh, h)
    y = sigmoid(y_)
    if i%100 == 0:
        plotx.append(len(plotx)+1)
        ploty.append((y_correct - y)**2)

    # 反向
    e = (y_correct - y) * y * (1-y)
    delta_wh = (np.dot(alpha * e, h)).T
    delta_wx = (np.matmul(np.multiply(np.dot(alpha * e, wh.T[:-1]), np.multiply(h[:-1], (1-h[:-1]))), x.T))

    wh = wh + delta_wh
    wx = wx + delta_wx

    return wx, wh


if __name__ == "__main__":
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    label = [0, 1, 1, 0]

    wx = create_w(2, 3)
    wh = create_w(1, 3)

    for i in range(10000):
        x = random.randint(0, 3)
        wx, wh = train(np.array(data[x]).reshape([2, 1]), label[x], wx, wh, i)

    for i in range(4):
        print("x=", data[i])
        print("y=", test(np.array(data[i]).reshape([2, 1]), wx, wh))
    plt.plot(plotx, ploty)
    plt.show()
