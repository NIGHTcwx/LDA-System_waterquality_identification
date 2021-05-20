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

x1 = data1[:, 4]
y1 = data1[:, 5]
z1 = data1[:, 6]

# 数据2

x2 = data2[:, 4]
y2 = data2[:, 5]
z2 = data2[:, 6]

# 数据3

x3 = data3[:, 4]
y3 = data3[:, 5]
z3 = data3[:, 6]


# 数据4

x4 = data4[:, 4]
y4 = data4[:, 5]
z4 = data4[:, 6]

# 数据5

x5 = data5[:, 4]
y5 = data5[:, 5]
z5 = data5[:, 6]


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
plt.title("五类图像的二阶颜色矩像素点分布图")
plt.show()




