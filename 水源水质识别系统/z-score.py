import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
np.set_printoptions(threshold=np.inf)  # 显示完整数据

# 加载数据
df_water = pd.read_csv('data.csv')  # 本地加载
x, y = df_water.iloc[:, 1:].values, df_water.iloc[:, 0].values  # 使用iloc函数，索引数据。显然，x为数据集，y为标签
# 数据的标准化
sc = StandardScaler()  # 引入StandardScaler函数，进行标准化 x'=(x-𝝁)/𝝈
data_std = sc.fit_transform(x)  # 对x_train数据集标准化后转化，便于计算，下同
print(data_std)
