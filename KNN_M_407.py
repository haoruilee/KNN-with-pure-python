"""
使用马式距离做KNN分类
输出正确率验证结果

61518407李浩瑞 2021.1.5
"""
from KNN_O_407 import *

def Grad(data,label,A):
    y=len(data[0])
    n=len(data)
    Prow = []
    """计算概率矩阵"""
    P=np.zeros((n, n))
    for i in range(n):
        sums=0
        #print("sums:",sums)
        for j in range(n):
            if j == i:P[i][j]=0
            else:
                for k in range(n):
                    if k!=i:sums+=np.exp(-np.square(M_distance(A,data[i],data[k])))
                    else:pass
                P[i][j]=np.exp(-np.square(M_distance(A,data[i],data[j])))
    """初始化"""
    for i in range(len(P)):
        temp=0
        for j in range(len(P[0])):
            if label[i]==label[j]:temp=temp+P[i][j]
        Prow.append(temp)
        print(Prow)
    """求梯度"""
    gradients = np.zeros((y, y))
    for i in range(len(data)):
        k_sum = np.zeros((y, y)) 
        k_same_sum = np.zeros((y, y))
        for k in range(len(data)):
            out_prod = np.outer(data[i] - data[k], data[i] - data[k])
            k_sum += P[i][k] * out_prod
            if label[k] == label[i]:k_same_sum += P[i][k] * out_prod
        gradients += Prow[i] * k_sum - k_same_sum
        print("gradients",gradients)
    gradients = 2 * np.dot(A,gradients)
    print("Finally gradients",gradients)
    return gradients


def calA(data,label,n):
    epoch=100
    y=len(data[0])
    #A=np.random.standard_normal(size=(n, y))
    A=[[4.89514866e-02,  9.67910746e-02, -1.23441573e+00, -2.08268957e+00,
        5.08443656e-02,  5.12681069e-03,  6.06616156e-03, -4.32789149e+00,
        2.68310777e-03,  3.75712687e-01,  3.58570518e-02, 2.27764534e-03,
        1.39429652e-01],
        [-2.26788622e+00, -1.34528932e+00, -5.10608630e-02, -3.43686243e-01,
        7.17562172e-02, -3.28447762e-01, -7.10408191e-03, -4.81660109e-02,
        2.75824997e-02, -4.20081216e+00, -1.76492057e-03,  1.30272212e-02,
        4.48312757e-02]]
    for i in range(epoch):
        grad=Grad(data,label,A)
        if (np.max(grad)<0.01):break
        A=A-grad*A*0.01
        print("epoch:{},A:{}".format(i,A))
    return A



def M_KNN(k):
    r = 0
    w = 0
    for i in range(len(valdata)):
        distance = []
        pred_label = []
        for j in range(len(data)):
            dist = M_distance(A, valdata[i], data[j])
            distance.append((label[j], dist))
        distance.sort(key=operator.itemgetter(1))
        temp = distance[0:k]
        for klabel in temp:pred_label.append(int(klabel[0]))
        pred = max(pred_label, key=pred_label.count)
        if pred == vallabel[i]:r += 1
        else:w += 1
    acc = r/(r+w)  # 返回正确率用于绘图
    return acc


if __name__ == '__main__':
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
    #填充空值，不填就算不了梯度
    Fillnan_and_Normal(data, valdata, label)
    print("data",data)
    print("label",label)
    #A=calA(data,label,2,learning_rate=0.001, precision=2, max_iters=100)
    # 使用上式计算出最优的A
    A = [[4.89514866e-02,  9.67910746e-02, -1.23441573e+00, -2.08268957e+00,
        5.08443656e-02,  5.12681069e-03,  6.06616156e-03, -4.32789149e+00,
        2.68310777e-03,  3.75712687e-01,  3.58570518e-02, 2.27764534e-03,
        1.39429652e-01],
        [-2.26788622e+00, -1.34528932e+00, -5.10608630e-02, -3.43686243e-01,
        7.17562172e-02, -3.28447762e-01, -7.10408191e-03, -4.81660109e-02,
        2.75824997e-02, -4.20081216e+00, -1.76492057e-03,  1.30272212e-02,
        4.48312757e-02]]
    x = range(1, 150)
    y = []
    for i in x:
        y.append(M_KNN(i))
    plt.plot(x, y)
    plt.xlabel('Number of points')
    plt.ylabel('Accuracy')
    plt.savefig('knn_2.png')
    print("Success! KNN_M results saved at knn_2.png!")
