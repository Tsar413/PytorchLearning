import numpy as np
import torch
import torchvision.transforms as transforms

data1 = np.array([
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
    [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
], dtype='uint8')

# 计算mean和std的方法
# 1. 将数据转化为C W H，并归一化到[0,1]
# N: 一个batch中的图像数量 C: Channel图像中的通道数 通道指RGB: 3，黑白: 1
# W: Width 图像水平维度的像素 H: Height 图像垂直维度的像素
# ToTensor 将数据X/255得到 并进行转化
data1 = transforms.ToTensor()(data1)
# 2. 对数据进行扩维，增加batch维度 图像最终变为NCWH
data1 = torch.unsqueeze(data1, 0)
# 3. 创建三维空列表
nb_samples = 0
channel_mean = torch.zeros(3)
channel_std = torch.zeros(3)
print(data1.shape)
N, C, W, H = data1.shape[:4]
# 将w,h维度的数据展平，为batch，channel,data,然后对三个维度上的数分别求和和标准差
data1 = data1.view(N, C, -1)
# 展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
channel_mean += data1.mean(2).sum(0)
# 展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
channel_std += data1.std(2).sum(0)
# 获取所有batch的数据，这里为1
nb_samples += N
# 获取同一batch的均值和标准差
channel_mean /= nb_samples
channel_std /= nb_samples
print(channel_mean, channel_std)

dataTransforms = transforms.Compose([
    transforms.Normalize(mean=channel_mean, std=channel_std)
])

data2 = np.array([
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
    [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
], dtype='uint8')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=channel_mean, std=channel_std)
])
data2 = transform(data2)
print(data2)
