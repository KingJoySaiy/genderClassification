import torch.nn as nn
from dataset.dataLoader import TrainData
from constant.constPath import *
from os.path import join
from torchvision import models


def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.sum(t.float())


def startTrain():
    data = TrainData()
    # net = ResNet50 if newModel else torch.load(modelPath)
    net = models.resnet50(pretrained=True) if newModel else torch.load(modelPath)
    if needCuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optm = torch.optim.Adam(net.parameters(), lr=learningRate)

    print('start training!')
    for i in range(trainEpochs):
        trainData, trainLabel = data.nextTrain()
        if needCuda:
            x = torch.from_numpy(trainData).float().cuda()
            y = torch.from_numpy(trainLabel).long().cuda()
        else:
            x = torch.from_numpy(trainData).float()
            y = torch.from_numpy(trainLabel).long()

        net.train()
        y_hat = net(x)
        loss = criterion(y_hat, y)
        optm.zero_grad()
        loss.backward()
        optm.step()
        print("Epoch:{}, Loss:{:.4f}".format(i + 1, loss.item()))

        # if (i + 1) % saveModelEpoch == 0:
        if loss.item() < 0.035:
            '''
            net.eval()
            accuracySum = 0
            validData, validLabel = data.nextValid()
            while validData is not None:
                if needCuda:
                    testIn = torch.from_numpy(validData).float().cuda()
                    testLab = torch.from_numpy(validLabel).long().cuda()
                else:
                    testIn = torch.from_numpy(validData).float()
                    testLab = torch.from_numpy(validLabel).long()

                accuracySum += test(net(testIn), testLab)
                validData, validLabel = data.nextValid()

            print("Accuracy:{:.0f}".format(accuracySum))
            modelName = "Loss{:.4f}_Accu{:.0f}.pkl".format(round(loss.item(), 3), round(float(accuracySum), 3))
            '''
            modelName = "Loss:{:.6f}.pkl".format(round(loss.item(), 6))
            torch.save(net, join('savedModel', modelName))

    torch.save(net, modelPath)


if __name__ == '__main__':
    setSeed()
    startTrain()
