import torch
import numpy as np

flag = 3

if flag == 1:  # 均值和标准差皆为张量
    mean = torch.arange(start=1, end=10, step=2, dtype=torch.float)
    std = torch.arange(start=1, end=10, step=2, dtype=torch.float)
    t_normal = torch.normal(mean=mean, std=std)
    print(t_normal)
elif flag == 2:  # 均值和标准差皆为标量 需要添加size
    t_normal = torch.normal(mean=1., std=4., size=(4,))
    print(t_normal)
elif flag == 3:  # 均值为张量, 标准差为标量
    mean = torch.arange(start=1, end=10, step=2, dtype=torch.float)
    std = 1
    t_normal = torch.normal(mean=mean, std=std)
    print(t_normal)
elif flag == 4:  # 均值为标量, 标准差为张量
    mean = 4
    std = torch.arange(start=1, end=10, step=2, dtype=torch.float)
    t_normal = torch.normal(mean=mean, std=std)
    print(t_normal)
