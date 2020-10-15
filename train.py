import torch
import torch.nn as nn
import numpy as np
from constant.dataset import *
from constant.constPath import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 50 * 50, 2)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        return self.out(x)


def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


def startTrain():
    data = TrainData(batchSize)

    # data, label = getTrainData()
    # n, t, x, y = data.shape
    print('start initializing!')

    # net = CNN()
    net = torch.load(modelPath)
    criterion = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss损失
    optm = torch.optim.Adam(net.parameters(), lr=learningRate)  # Adam优化
    epochs = 36  # 训练1000�?

    print('start training!')
    for i in range(epochs):
        if (i + 1) % 18 == 0:
            data.shuffle()
        trainData, trainLabel, validData, validLabel = data.nextTrainValid()

        # 指定模型为训练模式，计算梯度
        net.train()
        # 输入值都需要转化成torch的Tensor
        x = torch.from_numpy(trainData).float()
        y = torch.from_numpy(trainLabel).long()
        y_hat = net(x)
        # print(type(y), y.shape)
        # print(type(y_hat), y_hat.shape)

        loss = criterion(y_hat, y)  # 计算损失
        optm.zero_grad()  # 前一步的损失清零
        loss.backward()  # 反向传播
        optm.step()  # 优化

        # 指定模型为计算模�?
        net.eval()
        test_in = torch.from_numpy(validData).float()
        test_l = torch.from_numpy(validLabel).long()
        test_out = net(test_in)
        # 使用我们的测试函数计算准确率
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), accu))

    torch.save(net, modelPath)
