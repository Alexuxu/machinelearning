import numpy as np
import math
import time
from graphviz import Digraph


# 结点类
class Node:
    def __init__(self, code, data=0, dic=0):
        self.data = data
        self.code = code        # 结点的编码
        self.dic = dic
        self.isleaf = False

    def set_value(self, son, info):
        self.son = son          # 子节点信息，类型为（[子节点]，[对应类别]）
        self.info = info        # 结点信息（叶子结点：分类结果（str类型的结果）；非叶子结点：分类依据（dic的index））

    def set_leaf(self):
        self.isleaf = True


class DecisionTree:
    def __init__(self, dic):
        self.dic = dic
        self.con_or_dis = dict()
        self.node_head = 0

    def test(self, data):
        return self.node_test(data, self.node_head)

    def node_test(self, data, node):
        result = 0
        if node.isleaf:
            result = node.info
        else:
            # 如果是连续值
            if self.con_or_dis[node.info]:
                if float(data[node.info]) <= float(node.son[0][1][1:]):
                    result = self.node_test(data, node.son[0][0])
                else:
                    result = self.node_test(data, node.son[1][0])

            # 如果是离散值
            else:
                for i in node.son:
                    if data[node.info] == i[1]:
                        result = self.node_test(data, i[0])
                    else:
                        print("类型%s不存在"%data[node.info])
        return result


    def create_tree(self, data):
        dot = Digraph(comment="Decision Tree")

        node_to_be_divided = list()
        node_all = list()
        self.node_head = Node("1", data)
        node_to_be_divided.append(self.node_head)
        node_code = "2"

        dis_dic = dict()
        con_dic = dict()
        t_data = np.array(data).T.tolist()[:-1]
        for index, i in enumerate(t_data):
            if index not in self.con_or_dis:
                self.con_or_dis[index] = get_con_or_dis(i, 5)

        # 储存离散值的所有可能取值
        for i in self.con_or_dis:
            if not self.con_or_dis[i]:
                t = set()
                for j in t_data[i]:
                    t.add(j)
                dis_dic[i] = t

        # 储存连续值的所有均值
        for i in self.con_or_dis:
            if self.con_or_dis[i]:
                float_data = [float(x) for x in t_data[i]]
                sorted_data = sorted(float_data)
                t_average = [(x + y) / 2 for x, y in zip(sorted_data[1:], sorted_data[:-1])]
                con_dic[i] = t_average

        # 开始产生分支
        while True:

            # 如果待分支结点为空，停止分支
            if not node_to_be_divided:
                break

            # 当前准备分支的结点
            node_now = node_to_be_divided[0]

            # 若待分支结点的数据为相同类型，停止此次分支
            same, type = get_type(node_now.data)
            if same:
                dot.node(node_now.code, "%s" % type, fontname="SimHei")
                node_now.info = type
                node_now.isleaf = True
                node_to_be_divided.remove(node_now)
                continue

            max = float()  # 最大信息增益
            max_index = int()  # 最大信息增益对应的属性索引
            max_value = float()  # 最大信息增益对应的分割值（若最大增益为离散值则不需要）
            data_T = np.array(node_now.data).T.tolist()  # 数据转置

            for index, i in enumerate(data_T[:-1]):

                # 如果该属性为连续值
                if self.con_or_dis[index]:

                    data_list = [np.array(node_now.data).T.tolist()[-1], i]
                    data_list = np.array(data_list).T.tolist()  # 待求信息增益的数据列表

                    for j in con_dic[index]:
                        gain = get_gain_con(data_list, j)

                        if gain > max:
                            max = gain
                            max_index = index
                            max_value = j
                            max_con_or_dis = self.con_or_dis[index]

                else:
                    data_list = [np.array(node_now.data).T.tolist()[-1], i]
                    data_list = np.array(data_list).T.tolist()

                    gain = get_gain_dis(data_list)
                    if gain > max:
                        max = gain
                        max_index = index
                        max_con_or_dis = self.con_or_dis[index]

            # 绘制结点
            dot.node(node_now.code, "%s" % self.dic[max_index], fontname="SimHei")

            # 生成子节点数据
            # 如果最大增益为连续值
            if max_con_or_dis:
                list_node1_data = list()
                list_node2_data = list()
                for i in node_now.data:
                    if float(i[max_index]) <= max_value:
                        list_node1_data.append(i)
                    else:
                        list_node2_data.append(i)
            # 如果最大增益为离散值
            else:
                son_data_dic = dict()
                for i in dis_dic[max_index]:
                    if i not in son_data_dic:
                        son_data_dic[i] = list()

                for i in node_now.data:
                    son_data_dic[i[max_index]].append(i)

            # 生成子结点
            # 连续值
            if max_con_or_dis:
                node1 = Node(node_code, list_node1_data)
                node_code = str(int(node_code) + 1)
                node2 = Node(node_code, list_node2_data)
                node_code = str(int(node_code) + 1)

                son = [[node1, "≤%.3f" % max_value], [node2, "＞%.3f" % max_value]]
                node_now.set_value(son, max_index)
                node_to_be_divided.append(node1)
                node_to_be_divided.append(node2)
            # 离散值
            else:
                son = list()
                for i in son_data_dic:  # i为本次分支对应的属性值

                    # 如果此子结点数据为空,找到当前结点最大概率的结果
                    if not son_data_dic[i]:
                        max_type = dict()
                        max_num = 0
                        for index, j in enumerate(node_now.data[-1]):
                            if j not in max_type:
                                max_type[j] = 1
                                if max_type[j] > max_num:
                                    max_num = max_type[j]
                                    temp_data = list()
                                    temp_data.append(node_now.data[index])
                            else:
                                max_type[j] = max_type[j] + 1
                                if max_type[j] > max_num:
                                    max_num = max_type[j]
                                    temp_data = list()
                                    temp_data.append(node_now.data[index])
                        t = [Node(node_code, temp_data), i]
                        node_code = str(int(node_code) + 1)
                        son.append(t)
                        node_to_be_divided.append(t[0])

                    else:
                        t = [Node(node_code, son_data_dic[i]), i]
                        node_code = str(int(node_code) + 1)
                        son.append(t)
                        node_to_be_divided.append(t[0])
                node_now.set_value(son, max_index)

            node_to_be_divided.remove(node_now)
            node_all.append(node_now)

        # 建立结点间关系
        iterate(node_all[0], dot)
        dot.render('test.gv', view='False')


# 计算信息熵
def get_entropy(data):
    type_dict = dict()
    type_list = list()
    for i in data:
        if i not in type_dict:
            type_dict[i] = 1
        else:
            type_dict[i] = type_dict[i] + 1

    for i in type_dict:
        type_list.append(type_dict[i])

    temp = [(x / sum(type_list)) * math.log2(x / sum(type_list)) for x in type_list]
    entropy = -sum(temp)
    return entropy


# 计算连续值信息增益
def get_gain_con(data, threshold):
    # 输入数据格式[[种类名, 数值1],[种类名，数值2]……]
    t = list()
    for i in data:
        t.append(i[0])
    entropy_old = get_entropy(t)
    classone = list()
    classtwo = list()
    for i in data:
        if float(i[1]) <= float(threshold):
            classone.append(i[0])
        else:
            classtwo.append(i[0])
    co_one = len(classone)/(len(classone)+len(classtwo))
    co_two = len(classtwo)/(len(classone)+len(classtwo))
    entropy_new = co_one*get_entropy(classone) + co_two*get_entropy(classtwo)
    gain = entropy_old-entropy_new
    return gain


# 计算离散值信息增益
def get_gain_dis(data):
    # 输入数据格式[[种类名, 属性值1],[种类名，属性值2]……]
    # 求原信息熵
    data_t = np.array(data).T[0]
    entropy_old = get_entropy(data_t)

    # 求新信息熵
    length = len(data)
    entropy_new = float()
    data_dic = dict()
    for i in data:
        if i[1] not in data_dic:
            data_dic[i[1]] = list()
            data_dic[i[1]].append(i[0])
        else:
            data_dic[i[1]].append(i[0])

    for i in data_dic:
        entropy_new = entropy_new + (len(data_dic[i])/length)*get_entropy(data_dic[i])

    # 求信息增益
    gain = entropy_old - entropy_new
    return gain


def get_type(data):
    # 取关于类型的维度
    data_array = np.array(data).T.tolist()[-1]
    # 取第一个数据的类型，并与之后数据依次对比
    type = data_array[0]
    for i in data_array:
        if i != type:
            return False, 0
    return True, type


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


def iterate(node, dot):
    node.data = 0
    try:
        for i in node.son:
            code = i[0].code
            label = i[1]
            dot.edge(node.code, code, label, fontname="SimHei")
            iterate(i[0], dot)

    except AttributeError:
        return


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
    t = time.time()
    dic = {0: "花萼长度", 1: "花萼宽度", 2: "花瓣长度", 3: "花瓣宽度"}
    dic2 = {0: "色泽", 1: "根蒂", 2: "敲声", 3: "纹理", 4: "脐部", 5: "触感"}
    dic3 = {0: "色泽", 1: "根蒂", 2: "敲声", 3: "纹理", 4: "脐部", 5: "触感", 6: "密度", 7: "含糖率"}
    dic4 = {0: "长度", 1: "直径", 2: "高度", 3: "全重", 4: "去壳重", 5: "脏器重", 6: "壳重", 7: "环数"}
    data = import_data("abalone.txt")
    tree = DecisionTree(dic4)
    tree.create_tree(data)
    print("用时", time.time() - t, "秒")
