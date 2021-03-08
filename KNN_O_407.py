"""
使用欧式距离和马氏距离做KNN分类
输出正确率验证结果

61518407李浩瑞 2021.1.5
"""
import os,sys
import operator
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
#通过相对路径索引数据
project_path = os.getcwd()   # 获取当前文件路径的上一级目录
train_path = project_path+r"\data\train_data"  # 拼接训练路径字符串
test_path = project_path+r"\data\test_data.mat"  # 拼接测试路径字符串
val_path = project_path+r"\data\val_data.mat"

#读取数据
data = scio.loadmat(train_path)['data']
label = scio.loadmat(train_path)['label']
valdata = scio.loadmat(val_path)['data']
vallabel = scio.loadmat(val_path)['label']


def fullMean(data):
    """
    获得整个数据的均值
    return list(data[0])
    """
    lens=len(data[0])
    result = []
    number = [0]*lens
    s = [0]*lens
    for j in range(lens):
        for i in range(len(data)):
            if(np.isnan(data[i][j]) == False):
                number[j] += 1
                s[j] += data[i][j]
    for i in range(len(s)):result.append(s[i]/number[i])
    return result


def rowMean(data, label):
    """
    获得每个向量的均值
    return list(3, len(data[0]))
    """
    lens=len(data[0])
    result = np.zeros((3, lens))
    number = np.zeros((3, lens))
    for i in range(len(data)):
        for j in range(len(data[i])):
            if(np.isnan(data[i][j]) == False):
                ind = label[i]-1
                result[ind[0]][j] += data[i][j]
                number[ind[0]][j] += 1
    for i in range(len(result)):
        for j in range(len(result[i])):result[i][j] /= number[i][j]
    return result


def fillNan(data, label):
    """
    数据预处理：听过向量均值填充NAN
    return data
    """
    row_m = rowMean(data, label)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if(np.isnan(data[i][j])):data[i][j] = row_m[label[i][0]-1][j]
    return data


def Normalize(data, mean, var):
    """
    向量归一化
    void 修改data
    """
    for i in range(len(data)):
        for j in range(len(data[i])):data[i][j] = (data[i][j]-mean[j])/pow(var[j], 0.5)


def fullVar(data):
    """
    获得整个数据的方差
    return list(data[0])
    """
    lens=len(data[0])
    result = []
    mean = fullMean(data)
    s = [0]*lens
    for i in range(len(data)):
        for j in range(lens):
            t = label[i]
            s[j] += pow(data[i][j]-mean[j], 2)
    for i in range(len(s)):
        temp = s[i]/len(data)
        result.append(temp)
    return result


def Fillnan_and_Normal(train, test, label):
    fillNan(train, label)
    mean = fullMean(train)
    var = fullVar(train)
    Normalize(train, mean, var)
    Normalize(test, mean, var)


def O_distance(testInstance, trainInstance):
    """
    计算欧式距离
    return float
    """
    length = len(testInstance)
    distance = 0
    for i in range(length):
        testInstance1 = float(testInstance[i])
        trainInstance1 = float(trainInstance[i])
        distance += (np.square(testInstance1-trainInstance1))
    return math.sqrt(distance)

def M_distance(A, x1, x2):
    """
    调用欧式距离函数计算马氏距离
    return float
    """
    return O_distance(np.dot(A, x1), np.dot(A, x2))

def Prob(data, A):
    """
    计算概率矩阵
    return [len(data),len(data)]
    """
    n = len(data)
    P = np.zeros((n, n))
    for i in range(n):
        sums = 0
        for j in range(n):
            if j == i:
                P[i][j] = 0
            else:
                for k in range(n):
                    if k != i:
                        sums = sums + \
                            np.exp(-np.square(M_distance(A, data[i], data[k])))
                    else:
                        pass
                P[i][j] = np.exp(-np.square(M_distance(A, data[i], data[j])))
    return P


def KNN(k):
    """
    欧式距离KNN
    """
    r = 0
    w = 0
    for i in range(len(valdata)):
        distance = []
        pred_label = []
        for j in range(len(data)):
            D = O_distance(valdata[i], data[j])
            distance.append((label[j], D))
        distance.sort(key=operator.itemgetter(1))
        temp = distance[0:k]
        for klabel in temp:pred_label.append(int(klabel[0]))
        pred = max(pred_label, key=pred_label.count)
        if pred == vallabel[i]:r += 1
        else:w += 1
    acc = r/(r+w)  # 返回正确率用于绘图
    return acc

if __name__ == '__main__':
    #预处理：填充空值并归一化
    Fillnan_and_Normal(data, valdata, label)
    #检测不同的K效果
    x = range(1, 150)#取1-149的K
    y = []
    for i in x:
        y.append(KNN(i))
    plt.plot(x, y)
    plt.xlabel('Number of points')
    plt.ylabel('Accuracy')
    plt.savefig('knn_1.png')
    print("Success! KNN_O results saved at knn_1.png!")
