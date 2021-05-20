import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 数据
file1 = "data/1.txt"
file2 = "data/2.txt"
file3 = "data/3.txt"
file4 = "data/4.txt"
file5 = "data/5.txt"
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)
data3 = np.loadtxt(file3)
data4 = np.loadtxt(file4)
data5 = np.loadtxt(file5)

# 数据１

x1 = data1[:, 7]
y1 = data1[:, 8]
z1 = data1[:, 9]

# 数据2

x2 = data2[:, 7]
y2 = data2[:, 8]
z2 = data2[:, 9]

# 数据3

x3 = data3[:, 7]
y3 = data3[:, 8]
z3 = data3[:, 9]


# 数据4

x4 = data4[:, 7]
y4 = data4[:, 8]
z4 = data4[:, 9]

# 数据5

x5 = data5[:, 7]
y5 = data5[:, 8]
z5 = data5[:, 9]


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
plt.title("五类图像的三阶颜色矩像素点分布图")
plt.show()




