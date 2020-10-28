from dataset.fileReader import fileReader
from constant.constPath import *
import numpy as np


class TrainData:
    def __init__(self):
        # train: 1-trainSize, valid: trainSize-18000, shuffled
        idLabelSet = fileReader.getIdLabelSet()
        random.shuffle(idLabelSet)
        self.trainIdLabel = idLabelSet[:trainSize]
        self.validIdLabel = idLabelSet[trainSize:]
        self.trainLen, self.trainNow = trainSize, 0
        self.validLen, self.validNow = len(self.validIdLabel), 0

    # train: (trainBatch, 3, 224, 224) (trainBatch)
    def nextTrain(self):
        trainData = np.zeros((trainBatch, 3, imageH, imageW))
        trainLabel = np.zeros(trainBatch)
        ct = 0
        for name, label in self.trainIdLabel[self.trainNow:self.trainNow + trainBatch]:
            trainData[ct] = fileReader.readImage(join(trainImage, str(name) + '.jpg'))
            trainLabel[ct] = label
            ct += 1
        self.trainNow = ((self.trainNow + trainBatch) if (self.trainNow + 2 * trainBatch < self.trainLen) else 0)
        return trainData, trainLabel

    # # valid: (testBatch, 3, 224, 224) (testBatch)
    def nextValid(self):
        if self.validNow == self.validLen:
            self.validNow = 0
            return None, None
        nowLen = min(testBatch, self.validLen - self.validNow)
        validData = np.zeros((nowLen, 3, imageH, imageW))
        validLabel = np.zeros(nowLen)
        ct = 0
        for name, label in self.validIdLabel[self.validNow:self.validNow + nowLen]:
            validData[ct] = fileReader.readImage(join(trainImage, str(name) + '.jpg'))
            validLabel[ct] = label
            ct += 1
        res = validData, validLabel
        self.validNow += nowLen
        return res


class TestData:
    def __init__(self):
        self.idSet = fileReader.getIdSet()
        self.now = 0
        self.len = len(self.idSet)

    # test: (testBatch, 3, 224, 224) (testBatch, 1)
    def nextTest(self):
        if self.now == self.len:
            self.now = 0
            return None, None
        nowLen = min(testBatch, self.len - self.now)
        testData = np.zeros((nowLen, 3, imageH, imageW))
        ct = 0
        for i in self.idSet[self.now:self.now + nowLen]:
            testData[ct] = fileReader.readImageInitial(join(testImage, str(i) + '.jpg'))
            ct += 1
        res = testData, self.idSet[self.now:self.now + nowLen]
        self.now += nowLen
        return res
