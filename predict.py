import torch
import csv
from constant.dataset import TestData
from constant.constPath import modelPath, submitCSV, predictBatch


def startPredict():
    data = TestData()
    net = torch.load(modelPath)
    writer = csv.writer(open(submitCSV, "w+", newline=""))
    writer.writerow(['id', 'label'])

    ct = 0
    all = 5708 // predictBatch
    testData, id = data.nextTest()
    while testData is not None:
        print('predict Epoch: ', ct, '/', all)
        ct += 1
        test_in = torch.from_numpy(testData).float()
        test_out = net(test_in)
        label = test_out.max(-1)[1]
        for row in range(len(id)):
            writer.writerow([int(id[row]), int(label[row])])
        testData, id = data.nextTest()
