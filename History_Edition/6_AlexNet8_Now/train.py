import torch
import torch.nn as nn
from constant.dataset import TrainData
from constant.constPath import *


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=2),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


def startTrain():
    data = TrainData()

    # data, label = getTrainData()
    # n, t, x, y = data.shape
    print('start initializing!')

    if newModel:
        net = AlexNet()
    else:
        net = torch.load(modelPath)

    net.cuda()
    criterion = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss损失
    optm = torch.optim.SGD(net.parameters(), momentum=initialMomentum, lr=learningRate, weight_decay=weightDecay)
    epochs = trainEpochs

    print('start training!')
    for i in range(epochs):
        if i % oneTotal == 0:
            data.shuffle()
        trainData, trainLabel, validData, validLabel = data.nextTrainValid()

        # 指定模型为训练模式，计算梯度
        net.train()
        # 输入值都需要转化成torch的Tensor
        x = torch.from_numpy(trainData).float().cuda()
        y = torch.from_numpy(trainLabel).long().cuda()
        y_hat = net(x)
        # print(type(y), y.shape)
        # print(type(y_hat), y_hat.shape)

        loss = criterion(y_hat, y)  # 计算损失
        optm.zero_grad()  # 前一步的损失清零
        loss.backward()  # 反向传播
        optm.step()  # 优化

        net.eval()
        test_in = torch.from_numpy(validData).float().cuda()
        test_l = torch.from_numpy(validLabel).long().cuda()
        test_out = net(test_in)
        # 使用我们的测试函数计算准确率
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy:{:.2f}".format(i + 1, loss.item(), accu))

    torch.save(net, modelPath)
