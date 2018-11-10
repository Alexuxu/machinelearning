import numpy as np


def calcu_possi(data):
    sum = len(data)
    result = dict()
    for i in data:
        if i not in result:
            result[i] = 1 / sum
        else:
            result[i] = result[i] + 1 / sum
    return result


def bayes(data):
    classified_data = dict()
    for i in data:
        if i[-1] not in classified_data:
            t = list()
            t.append(i)
            classified_data[i[-1]] = t
        else:
            classified_data[i[-1]].append(i)

    classified_possi = dict()
    for i in classified_data:
        t = np.array(classified_data[i]).T.tolist()
        possi_list = list()
        for j in t[:-1]:
            possi_list.append(calcu_possi(j))

        classified_possi[i] = possi_list



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
    bayes(data)
