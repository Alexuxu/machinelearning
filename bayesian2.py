import numpy as np
import matplotlib.pyplot as plt
import random
import math


def calcu_possi(data):
    sum = len(data)
    result = dict()
    for i in data:
        if i not in result:
            result[i] = 1 / sum
        else:
            result[i] = result[i] + 1 / sum
    return result


# 获取正态分布的概率
def normal_possi(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


# 判断数据是离散还是连续，连续返回True，离散返回False
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
        self.data_dis_or_con = dict()       # 用于储存数据连续or离散，连续为True，离散为False
        for index, i in enumerate(t_data[:-1]):
            self.data_dis_or_con[index] = get_con_or_dis(i, 5)

        self.con_data = list()
        self.dis_data = list()

        # 储存数据的种类
        self.data_class = set()
        for i in t_data[-1]:
            if i not in self.data_class:
                self.data_class.add(i)

        # 将数据分为连续和离散
        for i in self.data_dis_or_con:
            if self.data_dis_or_con[i]:
                self.con_data.append(t_data[i])
            else:
                self.dis_data.append(t_data[i])

        # 数据中加入类别，求转置
        self.con_data.append(t_data[-1])
        self.con_data = np.array(self.con_data).T.tolist()
        self.dis_data.append(t_data[-1])
        self.dis_data = np.array(self.dis_data).T.tolist()

    # 计算离散值的先验概率
    def calculate_possibility(self):
        # 将数据分类
        classified_data = dict()
        for i in self.dis_data:
            if i[-1] not in classified_data:
                t = list()
                t.append(i)
                classified_data[i[-1]] = t
            else:
                classified_data[i[-1]].append(i)

        # 将数据依次求先验概率
        classified_possi = dict()
        for i in classified_data:
            t = np.array(classified_data[i]).T.tolist()
            possi_dict = dict()
            for j in t[:-1]:
                possi_dict = dict(possi_dict, **calcu_possi(j))

            classified_possi[i] = possi_dict

        # 求类别的先验概率
        clas_possi = calcu_possi(np.array(self.dis_data).T.tolist()[-1])

        # 储存所有属性可能取值
        values = set()
        for i in self.dis_data:
            for j in i[:-1]:
                if j not in values:
                    values.add(j)

        # 最终结果
        self.dis_dict = dict()
        for value in values:
            t_dict = dict()
            for clas, dic in zip(classified_possi.keys(), classified_possi.values()):
                if clas not in t_dict:
                    if value in dic:
                        t_dict[clas] = dic[value] * clas_possi[clas]
                    else:
                        t_dict[clas] = 0

            possi_sum = sum(t_dict.values())
            for i in t_dict:
                t_dict[i] = (t_dict[i] + 1) / (possi_sum + len(clas_possi))
            self.dis_dict[value] = t_dict
        print(self.dis_dict)

    # 假设连续值符合正态分布，计算正态分布的参数
    def calculate_parameter(self):
        # 将数据分类
        classified_data = dict()
        for i in self.con_data:
            if i[-1] not in classified_data:
                t = list()
                t.append(i)
                classified_data[i[-1]] = t
            else:
                classified_data[i[-1]].append(i)

        # 计算不同分类的正态分布参数
        self.con_dict = dict()
        for i in classified_data:
            t_data = classified_data[i]
            t_data = np.array(t_data).T.tolist()
            for index, j in enumerate(t_data[:-1]):
                out_dict = dict()           # 外层字典
                in_dict = dict()            # 内层字典
                in_dict["mu"] = np.mean(list(map(float, j)))
                in_dict["sigma"] = np.std(list(map(float, j)))
                out_dict[i] = in_dict
                if index not in self.con_dict:
                    self.con_dict[index] = dict()
                self.con_dict[index].update(out_dict)
        print(self.con_dict)

    def test(self, data):
        max_type = str()
        max_possi = 0
        for clas in self.data_class:
            possi_sum = 1
            dis_index = 0
            for index, i in enumerate(data):
                if self.data_dis_or_con[index]:
                    possi_sum *= normal_possi(float(i), self.con_dict[dis_index][clas]['mu'], self.con_dict[dis_index][clas]['sigma'])
                else:
                    possi_sum *= self.dis_dict[i][clas]
            if possi_sum > max_possi:
                max_possi = possi_sum
                max_type = clas

        return max_type


if __name__ == "__main__":
    data = import_data("watermelon.txt")
    bayes = BayesClassifier(data)
    bayes.calculate_possibility()
    bayes.calculate_parameter()

    for i in data:
        print(bayes.test(i[:-1]))
