import torch

# 自动求导
# 2023年10月23日

flag = 3

if flag == 0:
    w = torch.tensor(data=[1.], requires_grad=True)
    x = torch.tensor(data=[2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True) # 保留生成图
    print(w.grad)
elif flag == 2:
    w = torch.tensor(data=[1.], requires_grad=True)
    x = torch.tensor(data=[2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)
    y1 = torch.add(a, b)

    loss = torch.cat(tensors=[y0, y1], dim=0)
    grad_tensors = torch.tensor([1., 1.])  # 1 * 3 + 1 * 2 = 5

    loss.backward(gradient=grad_tensors)
    print(w.grad)
elif flag == 3:
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)  # y = x ^ 2

    grad1 = torch.autograd.grad(outputs=y, inputs=x, create_graph=True)
    print(grad1)  # tensor([6.])  第一次求导

    grad2 = torch.autograd.grad(outputs=grad1[0], inputs=x)
    print(grad2)  # tensor([2.]) 第二次求导

