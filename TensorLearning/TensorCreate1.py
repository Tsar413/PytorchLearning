import numpy as np
import torch

flag = 3

if flag == 1:
    data = 1.0

    torch.tensor(
        data,  # 数据
        dtype=None,  # 数据类型 和上方保持一致
        device=None,  # CPU或Cuda(GPU)
        requires_grad=False  # 是否需要计算梯度
    )
elif flag == 2:
    data = np.ones(3, 3)  # float64类型3*3矩阵
    print("data的数据类型为: " + data.dtype)
    t = torch.tensor(data)  # CPU
    # t = torch.tensor(data, device="cuda")  #  GPU
    print(t)
elif flag == 3:
    data = np.array([1, 2, 3], [4, 5, 6])

    t1 = torch.from_numpy(data)
    t2 = torch.tensor(data, device="cuda")
    print(t1)
elif flag == 4:  # 共享内存
    data = np.array([1, 2, 3], [4, 5, 6])
    t1 = torch.from_numpy(data)
    data[0, 0] = 4
    print("Torch: " + t1)
    print("Data: " + data)
else:
    data = np.array([1, 2, 3], [4, 5, 6])
    t1 = torch.from_numpy(data)
    t1[0, 0] = 5
    print("Torch: " + t1)
    print("Data: " + data)

