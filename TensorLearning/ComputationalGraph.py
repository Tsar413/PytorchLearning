import torch

# 计算图
# 2023年10月21日

w = torch.tensor(data=[1.], requires_grad=True)
x = torch.tensor(data=[2.], requires_grad=True)

a = torch.add(w, x)
# a.retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)
