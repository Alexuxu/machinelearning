import numpy as np


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))*2 - 1


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
        for i in data[1:]:
            result.append(i)
        result.append(data[0])
        data = np.array(result).T.tolist()

    return data


def get_con_or_dis(data, n):
    t_set = set()
    for i in data:
        try:
            t = float(i)
        except ValueError:
            return False
        else:
            t_set.add(i)
            if len(t_set)>n:
                return True
    return False


class Knn():

    def __init__(self, data):
        self.data = data.copy()
        t_data = np.array(self.data).T.tolist()
        self.con_or_dis_dict = dict()
        for index, i in enumerate(t_data[:-1]):
            self.con_or_dis_dict[index] = get_con_or_dis(i, 5)

        self.maxmin = list()
        for index, i in enumerate(t_data[:-1]):
            if self.con_or_dis_dict[index]:
                t_list = list()
                t_list.append(max(i))
                t_list.append(min(i))
                self.maxmin.append(t_list)

    def getdistance(self, data1, data2):
        distance = 0
        index = 0
        for i, j in zip(data1, data2):
            if self.con_or_dis_dict[index]:
                distance += sigmoid(np.sqrt(abs(float(i)**2-float(j)**2)))
            else:
                if i != j:
                    distance += 1
            index += 1

        return distance

    def test(self, data, k):
        t_data = self.data.copy()
        nearest_list = list()
        for index in range(k):
            nearest = 999
            nearest_data = list()
            for i in t_data:
                distance = self.getdistance(data, i[:-1])
                if distance < nearest:
                    nearest = distance
                    nearest_data = i
            nearest_list.append(nearest_data)
            t_data.remove(nearest_data)

        max_result = dict()
        max_value = 0
        result = str()
        for i in nearest_list:
            if i[-1] not in max_result:
                max_result[i[-1]] = 1
                if max_result[i[-1]] > max_value:
                    max_value = max_result[i[-1]]
                    result = i[-1]
            else:
                max_result[i[-1]] += 1
                if max_result[i[-1]] > max_value:
                    max_value = max_result[i[-1]]
                    result = i[-1]

        return result


if __name__ == "__main__":
    data = import_data("watermelon2.txt")
    knn = Knn(data)

    for i in data:
        print(knn.test(i[:-1], 5))
