import torch
import torch.nn as nn
import numpy as np
import readFile
from readFile import imageSize

data = readFile.getTrainData()
# print(data.shape)

trainSize = 15000

# 100, 120001
n, l = data.shape

"""
# 归一化
for j in range(l - 1):
    meanVal = np.mean(data[:, j])  # data[:, j] 第j列, mean()取均值, std()取标准差
    stdVal = np.std(data[:, j])
    data[:, j] = (data[:, j] - meanVal) / stdVal

np.random.shuffle(data)

# print(data[:1, :l - 1])
train_data = data[:trainSize, :l - 1]
train_lab = data[:trainSize, l - 1]
test_data = data[trainSize:, :l - 1]
test_lab = data[trainSize:, l - 1]


class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(SIZE, 2)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


net = LR()
criterion = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss损失
optm = torch.optim.Adam(net.parameters())  # Adam优化
epochs = 150  # 训练1000次

for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x = torch.from_numpy(train_data).float()
    y = torch.from_numpy(train_lab).long()
    y_hat = net(x)
    loss = criterion(y_hat, y)  # 计算损失
    optm.zero_grad()  # 前一步的损失清零
    loss.backward()  # 反向传播
    optm.step()  # 优化

    if (i + 1) % 1 == 0:  # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in = torch.from_numpy(test_data).float()
        test_l = torch.from_numpy(test_lab).long()
        test_out = net(test_in)
        # 使用我们的测试函数计算准确率
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), accu))
"""