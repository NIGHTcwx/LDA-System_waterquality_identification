import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import *
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 加载数据
df_water = pd.read_csv('data/data.csv')  # 本地加载

# 将数据划分为训练集和测试集，比例为：8:2
x, y = df_water.iloc[:, 1:].values, df_water.iloc[:, 0].values  # 使用iloc函数，索引数据。显然，x为数据集，y为标签
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# 数据的标准化
sc = StandardScaler()  # 引入StandardScaler函数，进行标准化 x'=(x-𝝁)/𝝈
x_train_std = sc.fit_transform(x_train)  # 对x_train数据集标准化后转化，便于计算，下同
x_test_std = sc.fit_transform(x_test)

# 计算均值向量
np.set_printoptions(precision=4)  # 保留四位小数
mean_vecs = []
for label in range(1, 6):  # 标签有1,2,3,4,5共五类
    mean_vecs.append(np.mean(x_train_std[y_train == label], axis=0))  # 将求好的均值向量append进mean_vecs这个列表
# print("Mean Vectors %s:" % label,mean_vecs[label-1])

# 计算类内散布矩阵
k = 9  # 数据共有九个分量
Sw = np.zeros((k, k))  # 建立九行九列的方阵，准备存储类内散布矩阵
for label, mv in zip(range(1, 6), mean_vecs):
    Si = np.zeros((k, k))
    Si = np.cov(x_train_std[y_train == label].T)  # 协方差
    Sw += Si
# print("类内散布矩阵：", Sw.shape[0], "*", Sw.shape[1])
# print("类内散布矩阵：\n", Sw)
# print("类内标签分布：",np.bincount(y_train)[1:])

# 计算类间散布矩阵
mean_all = np.mean(x_train_std, axis=0)  # 求得训练集的均值
Sb = np.zeros((k, k))  # 存放类间散布矩阵的九行九列的方阵
for i, col_mv in enumerate(mean_vecs):
    n = x_train[y_train == i + 1, :].shape[0]  # 求得每一类的样本个数
    col_mv = col_mv.reshape(k, 1)  # 列均值向量
    mean_all = mean_all.reshape(k, 1)  # 转化为列向量
    Sb += n * (col_mv - mean_all).dot((col_mv - mean_all).T)
# print("类间散布矩阵：", Sb.shape[0], "*", Sb.shape[1])
# print("类间散布矩阵：\n", Sb)

# 计算广义特征值
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))  # 使用np.linalg.eig函数求出散布矩阵的特征值和特征向量，将特征值存储到eigenvalues中，将特征向量存储到eigenvectors中，inv为取逆，dot为点乘
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]  # 对特征值取绝对值
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)  # 按降序对特征值进行排序
# print(eigen_pairs[0][1][:, np.newaxis].real)  # 第一特征向量
# print("特征值降序排列：")
# for eigenvalue in eigen_pairs:
#     print(eigenvalue[0])

# 线性判别捕捉，计算辨识力
tot = sum(eigenvalues.real)  # 计算九个分量的总特征值
discr = []
# discr=[(i/tot) for i in sorted(eigen_vals.real,reverse=True)]
for i in sorted(eigenvalues.real, reverse=True):
    discr.append(i / tot)
cum_discr = np.cumsum(discr)  # 计算累加方差
# name = ("R1", "G1", "B1", "R2", "G2", "B2", "R3", "G3", "B3")
# plt.bar(range(1, 10), discr, color=['r', 'g', 'b'], alpha=0.7, align='center', label='各阶颜色矩分量的辨识力', tick_label=name)
# plt.step(range(1, 10), cum_discr, where='mid', label='累加辨识力', color='#000000')
# plt.ylabel('辨识力')
# plt.xlabel('各阶颜色矩分量名')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc='best')
# plt.show()

# 转换矩阵
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real, eigen_pairs[2][1][:, np.newaxis].real))  # 水平方向铺开数组
# print(w)

# 样本数据投影到低维空间
x_train_lda = x_train_std.dot(w)
x_test_lda = x_test_std.dot(w)
# 数据提取
# 提取测试集和训练集进过投影后的一阶颜色矩
a1 = x_train_lda[:, 0]
a2 = x_train_lda[:, 1]
a3 = x_train_lda[:, 2]
b1 = x_test_lda[:, 0]
b2 = x_test_lda[:, 1]
b3 = x_test_lda[:, 2]
# 存储文件
np.savetxt('a1.txt', a1, fmt='%.3f', newline=os.linesep)
np.savetxt('a2.txt', a2, fmt='%.3f', newline=os.linesep)
np.savetxt('a3.txt', a3, fmt='%.3f', newline=os.linesep)
np.savetxt('b1.txt', b1, fmt='%.3f', newline=os.linesep)
np.savetxt('b2.txt', b2, fmt='%.3f', newline=os.linesep)
np.savetxt('b3.txt', b3, fmt='%.3f', newline=os.linesep)

# 读取分类后的数据
# 训练集
# file1 = "data/train_1.txt"
# file2 = "data/train_2.txt"
# file3 = "data/train_3.txt"
# file4 = "data/train_4.txt"
# file5 = "data/train_5.txt"
# 测试集
file1 = "data/test_1.txt"
file2 = "data/test_2.txt"
file3 = "data/test_3.txt"
file4 = "data/test_4.txt"
file5 = "data/test_5.txt"
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)
data3 = np.loadtxt(file3)
data4 = np.loadtxt(file4)
data5 = np.loadtxt(file5)

# 数据１
x1 = data1[:, 1]
y1 = data1[:, 2]
z1 = data1[:, 3]

# 数据2

x2 = data2[:, 1]
y2 = data2[:, 2]
z2 = data2[:, 3]

# 数据3

x3 = data3[:, 1]
y3 = data3[:, 2]
z3 = data3[:, 3]


# 数据4

x4 = data4[:, 1]
y4 = data4[:, 2]
z4 = data4[:, 3]

# 数据5

x5 = data5[:, 1]
y5 = data5[:, 2]
z5 = data5[:, 3]

# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1, y1, z1, c='r', label='第一类点')
ax.scatter(x2, y2, z2, c='g', label='第二类点')
ax.scatter(x3, y3, z3, c='b', label='第三类点')
ax.scatter(x4, y4, z4, c='y', label='第四类点')
ax.scatter(x5, y5, z5, c='m', label='第五类点')

# 绘制图例
ax.legend(loc='best')

# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('R', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('G', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('B', fontdict={'size': 15, 'color': 'red'})

# 展示
plt.title("LDA模型-测试集的分类散点图")
plt.show()

