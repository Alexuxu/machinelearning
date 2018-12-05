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


def normal_possi(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


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


class BayesClassifier:

    def __init__(self, data):
        self.data = data

        t_data = np.array(data).T.tolist()
        self.data_dis_or_con = dict()
        for index, i in enumerate(t_data[:-1]):
            self.data_dis_or_con[index] = get_con_or_dis(i, 5)

        self.con_data = list()
        self.dis_data = list()

        self.data_class = set()
        for i in t_data[-1]:
            if i not in self.data_class:
                self.data_class.add(i)

        for i in self.data_dis_or_con:
            if self.data_dis_or_con[i]:
                self.con_data.append(t_data[i])
            else:
                self.dis_data.append(t_data[i])

        self.con_data.append(t_data[-1])
        self.con_data = np.array(self.con_data).T.tolist()
        self.dis_data.append(t_data[-1])
        self.dis_data = np.array(self.dis_data).T.tolist()

    def calculate_possibility(self):
        t_data = np.array(self.dis_data).T.tolist()
        p_dict = dict()
        for index, i in enumerate(t_data[:-1]):
            t = set()
            for j in i:
                t.add(j)
            p_dict[index] = list(t)

        classified_data = dict()
        for i in self.dis_data:
            if i[-1] not in classified_data:
                t = list()
                t.append(i)
                classified_data[i[-1]] = t
            else:
                classified_data[i[-1]].append(i)

        classified_possi = dict()
        for i in classified_data:
            t = np.array(classified_data[i]).T.tolist()
            possi_dict = dict()
            for index, j in enumerate(t[:-1]):
                j.extend(p_dict[index])
                possi_dict = dict(possi_dict, **calcu_possi(j))

            classified_possi[i] = possi_dict

        clas_possi = calcu_possi(np.array(self.dis_data).T.tolist()[-1])

        values = set()
        for i in self.dis_data:
            for j in i[:-1]:
                if j not in values:
                    values.add(j)

        self.dis_dict = dict()
        for value in values:
            t_dict = dict()
            for clas, dic in zip(classified_possi.keys(), classified_possi.values()):
                if clas not in t_dict:
                    t_dict[clas] = dic[value]
            self.dis_dict[value] = t_dict

    def calculate_parameter(self):
        classified_data = dict()
        for i in self.con_data:
            if i[-1] not in classified_data:
                t = list()
                t.append(i)
                classified_data[i[-1]] = t
            else:
                classified_data[i[-1]].append(i)

        self.con_dict = dict()
        for i in classified_data:
            t_data = classified_data[i]
            t_data = np.array(t_data).T.tolist()
            for index, j in enumerate(t_data[:-1]):
                out_dict = dict()
                in_dict = dict()
                in_dict["mu"] = np.mean(list(map(float, j)))
                in_dict["sigma"] = np.sqrt(np.var(list(map(float, j)), ddof=1))
                out_dict[i] = in_dict
                if index not in self.con_dict:
                    self.con_dict[index] = dict()
                self.con_dict[index].update(out_dict)

    def test(self, data):
        max_type = str()
        max_possi = 0
        for clas in self.data_class:
            possi_sum = 1
            dis_index = 0
            for index, i in enumerate(data):
                if self.data_dis_or_con[index]:
                    t = normal_possi(float(i), self.con_dict[dis_index][clas]['mu'], self.con_dict[dis_index][clas]['sigma'])
                    possi_sum *= t
                    dis_index += 1
                else:
                    t = self.dis_dict[i][clas]
                    possi_sum *= t
            if possi_sum > max_possi:
                max_possi = possi_sum
                max_type = clas

        return max_type


if __name__ == "__main__":
    data = import_data("iris.txt")
    bayes = BayesClassifier(data)
    bayes.calculate_possibility()
    bayes.calculate_parameter()

    for i in data:
        print(bayes.test(i[:-1]))
