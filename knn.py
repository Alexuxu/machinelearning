import numpy as np

def getdistance(data):


def import_data(filename, reverse=False):
    f = open(filename, 'r', encoding='utf-8')
    data_str = f.read()
    data_t = data_str.split('\n')
    data = list()
    for index in data_t:
        t = index.split(',')
        data.append(t)

    if reverse:
        data = np.array(data).T.tolist()
        result = list()
        result.append(data[-1])
        for i in data[:-1]:
            result.append(i)
        data = np.array(result).T.tolist()

    return data


if __name__ == "__main__":
    data = import_data("watermelon.txt")