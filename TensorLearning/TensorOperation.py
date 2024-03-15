import torch
import numpy as np

flag = 9

if flag == 1:
    t1 = torch.ones(size=(3, 3))
    t2 = torch.full_like(input=t1, fill_value=5)
    t1_2_1 = torch.cat(tensors=[t1, t2], dim=0)
    t1_2_2 = torch.cat(tensors=[t1, t2, t1], dim=1)
    print(t1_2_1, t1_2_1.shape)
    print(t1_2_2, t1_2_2.shape)
elif flag == 2:
    t1 = torch.ones(size=(2, 3))
    t2 = torch.full_like(input=t1, fill_value=5)
    t1_2 = torch.stack(tensors=[t1, t2], dim=0)
    print(t1_2, t1_2.shape)
elif flag == 3:
    data = torch.ones(size=(2, 5))
    list_of_tensors = torch.chunk(input=data, chunks=2, dim=1)
    for t in list_of_tensors:
        print(t, t.shape)
elif flag == 4:
    data = torch.ones(size=(2,5))
    list_of_tensors = torch.split(tensor=data, split_size_or_sections=2, dim=1)
    for t in list_of_tensors:
        print(t, t.shape)
elif flag == 5:
    data = torch.ones(size=(2, 5))
    list_of_tensors = torch.split(tensor=data, split_size_or_sections=[2, 1, 2], dim=1)
    for t in list_of_tensors:
        print(t, t.shape)
elif flag == 6:
    data = torch.ones(size=(3, 3))
    idx = torch.tensor(data=[0, 2], dtype=torch.long)
    t_list = torch.index_select(input=data, index=idx, dim=1)
    print(t_list)
elif flag == 7:
    data = torch.range(start=0, end=9, step=2)
    judge = data.ge(5)
    t = torch.masked_select(input=data, mask=judge)
    print(t)
elif flag == 8:
    data = torch.randint(low=0, high=9, size=(1, 9), dtype=torch.float)
    t = torch.reshape(input=data, shape=(3, 3))
    print(t)
elif flag == 9:
    data = torch.randint(low=0, high=9, size=(2, 3, 3), dtype=torch.float)
    print(data)
    t = torch.transpose(input=data, dim0=0, dim1=2)
    print(t)

