import random


def sign(x):
    if x > 0:
        return 1
    else:
        return 0


def test(w, b, x):
    y = 0
    for i, j in zip(x, w):
        y = y + i*j

    return sign(y)


def train(w, b, x, y_correct):
    y = 0
    alpha = 0.01
    for i, j in zip(x, w):
        y = y + i*j
    y = y + b
    e = y_correct - sign(y)

    delta_y = list()
    for index, i in enumerate(w):
        w[index] = e*alpha*i
    b = e*alpha

    return w, b


if __name__ == "__main__":
    x = [[1, 1], [1, 0], [0, 1], [0, 0]]
    y = [1, 0, 0, 0]
    rand = random.randint(0, 3)
