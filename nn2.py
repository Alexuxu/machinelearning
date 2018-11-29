import r
import math
import numpy as np


# 随机产生系数矩阵
def create_w(m, n):
    w = list()
    for i in range(m):
        t = list()
        w.append(t)
        for j in range(n):
            t.append(0.002 * r.randint(0, 1000) - 1)
    return np.array(w)


class Lay:
    def __init__(self, num):
        self.num = num
        self.data = create_w(num, 1)
        self.next = 0

    def set_next(self, next):
        self.next = next


class Connection:
    def __init__(self, input, output):
        self


class NetWork:
    def __init__(self, structure):
        for i in structure:
            pass
