import torch
import torch.nn as nn
from constant.dataset import TrainData
from constant.constPath import *


def Conv3x3BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class VGGNet(nn.Module):
    def __init__(self):  # (trainSize, 3, 224, 224)
        super(VGGNet, self).__init__()
        block_nums = [2, 2, 3, 3, 3]  # vgg16
        # block_nums = [2, 2, 4, 4, 4]  # vgg19
        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=2)
        )

        self._init_params()

    @staticmethod
    def _make_layers(in_channels, out_channels, block_num):
        layers = [Conv3x3BNReLU(in_channels, out_channels)]
        for i in range(1, block_num):
            layers.append(Conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
        

def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


def startTrain():
    data = TrainData()
    print('start initializing!')

    if newModel:
        net = VGGNet()
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
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy:{:.2f}".format(i + 1, loss.item(), accu))

    torch.save(net, modelPath)
