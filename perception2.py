import random
import numpy as np


def sgn(x):
    for index_i, i in enumerate(x):
        for index_j, j in enumerate(i):
            if j > 10:
                x[index_i][index_j] = 1
            else:
                x[index_i][index_j] = 0
    return x


def mul(x, y):
    t = list()
    for i, j in zip(x, y):
        pass


def create_mat(m, n):
    result = []
    for i in range(m):
        t = []
        for j in range(n):
            t.append(random.random())
        result.append(t)
    return np.array(result)


def test(x, w1, w2, b1, b2):
    h = sgn(np.matmul(w1, x.T) + b1)
    y = np.matmul(w2, h) + b2

    return sgn(y)


def train(x, y_c, w1, w2, b1, b2):
    alpha = 0.1
    h = sgn(np.matmul(w1, x.T) + b1)
    y = np.matmul(w2, h) + b2

    e = y_c - sgn(y)
    delta_w2 = alpha * e * w2
    w2 = w2 + delta_w2
    b2 = b2 + alpha * e*0.1

    ew = e * w2.T
    b1 = b1 + ew * alpha * 0.001

    return w1, w2, b1, b2


if __name__ == "__main__":
    x_data = [[[0, 0]], [[1, 0]], [[0, 1]], [[1, 1]]]
    y_data = [[1], [0], [0], [1]]

    w1 = create_mat(2, 2)
    w2 = create_mat(1, 2)
    b1 = create_mat(2, 1)
    b2 = create_mat(1, 1)

    for i in range(10000):
        index = random.randint(0, 3)
        w1, w2, b1, b2 = train(np.array(x_data[index]), np.array(y_data[index]), w1, w2, b1, b2)

    # for i in range(4):
    #     print("x=", x_data[i])
    #     print("y=", int(test(np.array(x_data[i]), w1, w2, b1, b2)))
    # print("w1=", w1)
    # print("w2=", w2)
    # print("b1=", b1)
    # print("b2=", b2)
