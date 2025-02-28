from math import inf
import torch
import torch.nn as nn
from constant.dataset import TrainData
from constant.constPath import *
from os.path import join
import sys
from torchvision import datasets, models, transforms

'''
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, expansion=4):
        super(ResNet50, self).__init__()

        blocks = [3, 4, 6, 3]  # ResNet-50
        # blocks = [3, 4, 23, 3]  # ResNet-101
        # blocks = [3, 8, 36, 3]  # ResNet-152
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = [Bottleneck(in_places, places, stride, downsampling=True)]
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''

def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


def startTrain():
    data = TrainData()
    print('start initializing!')

    net = models.resnet152(pretrained=True) if newModel else torch.load(modelPath)
    if needCuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss损失
    # optm = torch.optim.SGD(net.parameters(), momentum=initialMomentum, lr=learningRate, weight_decay=weightDecay)
    optm = torch.optim.Adam(net.parameters(), lr=learningRate)
    oneTotal = imageTotal / trainBatch

    print('start training!')
    nowLoss = 0
    for i in range(trainEpochs):
        if i % oneTotal == 0:
            data.shuffle()
        trainData, trainLabel, validData, validLabel = data.nextTrainValid()

        # 指定模型为训练模式，计算梯度
        net.train()
        # 输入值都需要转化成torch的Tensor
        if needCuda:
            x = torch.from_numpy(trainData).float().cuda()
            y = torch.from_numpy(trainLabel).long().cuda()
            test_in = torch.from_numpy(validData).float().cuda()
            test_l = torch.from_numpy(validLabel).long().cuda()
        else:
            x = torch.from_numpy(trainData).float()
            y = torch.from_numpy(trainLabel).long()
            test_in = torch.from_numpy(validData).float()
            test_l = torch.from_numpy(validLabel).long()

        y_hat = net(x)
        loss = criterion(y_hat, y)  # 计算损失
        optm.zero_grad()  # 前一步的损失清零
        loss.backward()  # 反向传播
        optm.step()  # 优化

        net.eval()
        test_out = net(test_in)
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy:{:.2f}".format(i + 1, loss.item(), accu))
        
        if loss.item() < 0.05:
            torch.save(net, join('model', 'model-1.pkl'))
            sys.exit(0)
        
        nowLoss += loss.item()
        if (i + 1) % oneTotal == 0:
            torch.save(net, join('model', 'model' + str(nowLoss) + '.pkl'))
            print('model ' + str(nowLoss) + ' saved')
            nowLoss = 0

    torch.save(net, modelPath)
