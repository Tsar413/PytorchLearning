import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#  线性回归模型学习
#  2023年10月20日


# 学习率
lr = 0.1

x = torch.rand(size=(20, 1)) * 10
y = 2 * x + (5 + torch.randn(size=(20, 1)))

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for i in range(1000):
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播 求导
    loss.backward()

    # 更新数值 获得梯度gard
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 绘图
    if i % 20 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(i, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break
    plt.show()
