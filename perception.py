import random


def sign(x):
    if x > 0.5:
        return 1
    else:
        return 0


def test(w, b, x):
    y = 0
    for i, j in zip(x, w):
        y = y + i*j

    return y


def train(w, b, x, y_correct):
    y = 0
    alpha = 1e-10
    for i, j in zip(x, w):
        y = y + i*j
    y = y + b
    e = y_correct - y

    delta_y = list()
    for index, i in enumerate(w):
        w[index] = w[index] + e*alpha*i
    b = b + e*alpha

    return w, b


if __name__ == "__main__":
    x = [[1, 1], [1, 0], [0, 1], [0, 0]]
    y = [1, 1, 1, 0]
    w = [1, 0.5]
    b = 0.5
    rand = random.randint(0, 3)
    for i in range(5000):
        train(w, b, x[rand], y[rand])

    for i in range(4):
        print("x=", x[i])
        print("y=", test(w, b, x[i]))
