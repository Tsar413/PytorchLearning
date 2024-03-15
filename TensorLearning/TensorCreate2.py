import torch
import numpy as np

flag = 6

if flag == 1:
    torch.zeros(  # 依照size创建全0张量
        size=(3, 3, 3),  # 张量的形状，如（3，3），（3，224，224）
        out=None,  # 输出的张量
        device=None,  # CPU/GPU
        layout=torch.strided,  # 内存中布局形式
        requires_grad=None  # 是否需要梯度
    )
elif flag == 2:
    out_t = torch.tensor([1])
    t = torch.zeros(size=(3, 3), out=out_t)  # out的作用是把值赋给等号后的张量
    print(out_t, "\n")
    print(t)
elif flag == 3:
    data = np.array([[1, 2, 3], [3, 4, 5]])
    data1 = torch.from_numpy(data)
    t = torch.zeros_like(input=data1)  # input必须是torch，不能是numpy
    print(t)
elif flag == 4:
    data = np.ones(shape=(3, 2))
    t = torch.tensor(data)
    t1 = torch.full_like(input=t, fill_value=5)
    t2 = torch.full(size=(2, 2), fill_value=2)
    print(t1)
    print(t2)
elif flag == 5:
    t = torch.arange(start=1, end=10, step=2)
    print(t)
elif flag == 6:
    t = torch.linspace(start=1, end=11, steps=5)
    print(t)


