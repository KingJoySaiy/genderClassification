import torch
import torch.nn as nn
from dataset.dataLoader import TrainData
from constant.constPath import *
from os.path import join
from torchvision import models
from myModel import ResNet50


def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


def startTrain():
    data = TrainData()
    # net = ResNet50 if newModel else torch.load(modelPath)
    net = models.resnet50(pretrained=True) if newModel else torch.load(modelPath)
    if needCuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    # optm = torch.optim.SGD(net.parameters(), momentum=initialMomentum, lr=learningRate, weight_decay=weightDecay)
    optm = torch.optim.Adam(net.parameters(), lr=learningRate)
    oneTotal = imageTotal / trainBatch
    nowLoss, nowAccu = 0, 0

    print('start training!')
    for i in range(trainEpochs):
        trainData, trainLabel = data.nextTrain()
        validData, validLabel = data.nextValid()
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

        net.train()
        y_hat = net(x)
        loss = criterion(y_hat, y)
        optm.zero_grad()
        loss.backward()
        optm.step()

        net.eval()
        test_out = net(test_in)
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy:{:.2f}".format(i + 1, loss.item(), accu))
        '''
        if loss.item() < 0.04:
            torch.save(net, join('savedModel', 'savedModel-1.pkl'))
            sys.exit(0)
        '''
        nowLoss += loss.item()
        nowAccu += accu
        if (i + 1) % (oneTotal / 3) == 0:
            torch.save(net, join('savedModel', 'loss' + str(round(nowLoss, 3)) + '_accu' + str(round(float(nowAccu), 3)) + '.pkl'))
            print(str(round(nowLoss, 3)) + '_' + str(round(float(nowAccu), 3)) + '_saved')
            nowLoss = 0
            nowAccu = 0

    torch.save(net, modelPath)


if __name__ == '__main__':
    setSeed()
    startTrain()
