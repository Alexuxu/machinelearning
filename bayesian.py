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
    # 将数据分类
    classified_data = dict()
    for i in data:
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
    clas_possi = calcu_possi(np.array(data).T.tolist()[-1])

    # 储存所有属性可能取值
    values = set()
    for i in data:
        for j in i[:-1]:
            if j not in values:
                values.add(j)

    # 最终结果
    final_dict = dict()
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
            t_dict[i] = (t_dict[i]+1)/(possi_sum+len(clas_possi))
        final_dict[value] = t_dict
    print(final_dict)


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
