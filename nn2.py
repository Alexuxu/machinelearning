import random
import math
import numpy as np


# 随机产生系数矩阵
def create_w(m, n):
    w = list()
    for i in range(m):
        t = list()
        w.append(t)
        for j in range(n):
            t.append(0.002 * random.randint(0, 1000) - 1)
    return np.array(w)


