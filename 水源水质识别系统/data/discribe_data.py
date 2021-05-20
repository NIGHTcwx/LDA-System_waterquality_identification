import numpy as np
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import matplotlib.font_manager as mpt
from scipy.stats import norm
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

file = "data/data.txt"
a = np.loadtxt(file)
data = a[:, 9]

mu = data.mean()  # 计算均值
std = data.std()  # 计算标准差
print(mu, std)


fig = plt.figure()
ax = fig.add_subplot()
bins = 50
ax.hist(data, bins, range=(-0.02, 0.02), color='black', alpha=0.5)
# y = norm.pdf(bins, mu, sigma)
# plt.plot(bins, y, 'r--')
plt.xlabel('B通道三阶颜色矩值')
plt.ylabel('数据量')
plt.title(r'B通道三阶颜色矩直方图: $\mu=0.00106$, $\sigma=0.00969$')
plt.subplots_adjust(left=0.15)
plt.show()
