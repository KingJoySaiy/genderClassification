from constant.dataset import *
from train import startTrain
from constant.constPath import *
import csv
import torch


def startPredict(bestModelPath):
    data = TestData()
    net = torch.load(bestModelPath)

    if needCuda:
        net.cuda()
    with open(submitCSV, 'w+', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])

        ct = 0
        all = 5708 // predictBatch
        testData, id = data.nextTest()
        while testData is not None:
            print('predict Epoch: ', ct, '/', all)
            ct += 1
            net.eval()
            test_in = (torch.from_numpy(testData).float().cuda() if needCuda else torch.from_numpy(testData).float())
            test_out = net(test_in)
            label = test_out.max(-1)[1]
            for row in range(len(id)):
                writer.writerow([int(id[row]), int(label[row])])
            testData, id = data.nextTest()


# setSeed(globalSeed)


if needTrain:
    startTrain()
if needPredict:
    startPredict(join('model', 'model-1.pkl'))
