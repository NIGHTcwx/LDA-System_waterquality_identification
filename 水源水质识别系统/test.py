import numpy as np
import os
import re
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)  # 显示完整数据

# 1、图像的名字提取和图像切割
# 2、提取图像的rgb数据和类别标签
path = './images/'  # 得到需要处理的images图像的路径


def get_image_names(path=path):
    """
    此函数用来得到images里面的所有图像名字，最后存储到image_names列表里面。
    :param path:images
    :return:image_name
    """
    image_names = []  # 定义一个空列表，将函数返回值放入其中
    names = os.listdir(path)
    for i in names:
        if re.findall('^\\d_\\d+\\.jpg', i):  # 运用正则判断，符合\d_\d+\.jpg的文件名都会放入到image_names列表里面
            image_names.append(i)
    return image_names


def var(x=None):
    """
    求图像的三阶颜色矩，
    :param x: 图像的像素值矩阵
    :return:图像的三阶颜色矩
    """
    mid = np.mean(((x - x.mean()) ** 3))
    return np.sign(mid) * abs(mid) ** (1 / 3)  # 其中sign(mid)当mid<0时为-1，mid>0时为1


# 将图像的数据批量处理，为求解一、二、三阶颜色矩做好准备
image_names = get_image_names(path)  # 通过get_image_names函数得到所有图像的名字，存储到image_names列表里面
n = len(image_names)  # 图像的名字唯一，则image_names列表的长度可以代表图像的个数
image_data = np.zeros([n, 9])  # n行代表有n个数据，9列代表每个数据的rgb
image_labels = np.zeros([n])  # 存储n个样本的标签

for i in range(n):  # 遍历全部图像
    image = Image.open(path+image_names[i])  # 从get_image-names函数的返回值image_names里面读取第i个图片
    M, N = image.size  # M即为图片的长，N为图片的宽
    image = image.crop((M/2-50, N/2-50, M/2+50, N/2+50))  # 切割第i个图像，按以图像中心点为中心，边长为100的正方形切割图像
    r, g, b = image.split()  # 使用split函数将image分为r，g，b三通道
    rd = np.asarray(r)/255  # 转化成数组数据
    gd = np.asarray(g)/255
    bd = np.asarray(b)/255

    image_data[i, 0] = rd.mean()  # 一阶颜色矩
    image_data[i, 1] = gd.mean()
    image_data[i, 2] = bd.mean()

    image_data[i, 3] = rd.std()   # 二阶颜色矩
    image_data[i, 4] = gd.std()
    image_data[i, 5] = bd.std()

    image_data[i, 6] = var(rd)    # 三阶颜色矩
    image_data[i, 7] = var(gd)
    image_data[i, 8] = var(bd)

    image_labels[i] = image_names[i][0]  # 样本标签

# 建立模型部分
# 将数据拆分为80%的训练集和20%的测试集
image_data_train, image_data_test, image_labels_train, image_labels_test = train_test_split(image_data, image_labels, test_size=0.2, random_state=10)
lda = LinearDiscriminantAnalysis()  # 导入LDA算法
lda.fit(image_data_train, image_labels_train)  # 导入训练集来训练模型
# 水质评价
predict_test = lda.predict(image_data_test)  # 导入测试集来得到预测
print('测试集水质评价结果为：')
print(predict_test)  # 展示测试集的预测结果
confusion_matrix_test = confusion_matrix(image_labels_test, predict_test)  # 把得到的预测结果按labels划分得到混淆矩阵
print('混淆矩阵为：')
print(confusion_matrix_test)  # 展示混淆矩阵
print('预测的准确率为：')
print(accuracy_score(image_labels_test, predict_test))  # 展示准确率
image_data_test_new = lda.transform(image_data_test)  # transform测试集
plt.scatter(image_data_test_new[:, 0], image_data_test_new[:, 1], marker='+', c=image_labels_test)  # 以一阶r颜色矩为x轴，以一阶g颜色矩为y轴，'+'代表点，总共分为五类
plt.show()

















