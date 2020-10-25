import random
from dataset.readFile import *


class TrainData:
    def __init__(self):
        # train: 1-12600, valid: 12601-18000, shuffled
        idLabelSet = getIdLabelSet()
        random.shuffle(idLabelSet)
        self.trainIdLabel = idLabelSet[:int(imageTotal * trainProportion)]
        self.validIdLabel = idLabelSet[int(imageTotal * trainProportion):]
        self.trainLen, self.trainNow = len(self.trainIdLabel), 0
        self.validLen, self.validNow = len(self.validIdLabel), 0

    # train: (trainBatch, 3, 224, 224) (trainBatch)
    def nextTrain(self):
        trainData = np.zeros((trainBatch, 3, imageH, imageW))
        trainLabel = np.zeros(trainBatch)
        ct = 0
        for name, label in self.trainIdLabel[self.trainNow:self.trainNow + trainBatch]:
            trainData[ct] = readImage(join(trainImage, str(name) + '.jpg'))
            trainLabel[ct] = label
            ct += 1
        self.trainNow = ((self.trainNow + trainBatch) if (self.trainNow + 2 * trainBatch < self.trainLen) else 0)
        return trainData, trainLabel

    # # valid:(validBatch, 3, 224, 224) (validBatch)
    def nextValid(self):
        validData = np.zeros((validBatch, 3, imageH, imageW))
        validLabel = np.zeros(validBatch)
        ct = 0
        for name, label in self.validIdLabel[self.validNow:self.validNow + validBatch]:
            validData[ct] = readImage(join(trainImage, str(name) + '.jpg'))
            validLabel[ct] = label
            ct += 1
        self.validNow = ((self.validNow + validBatch) if (self.validNow + 2 * validBatch < self.validLen) else 0)
        return validData, validLabel


class TestData:
    def __init__(self):
        self.idSet = getIdSet()
        self.now = 0
        self.len = len(self.idSet)

    # test(testLen, 1, 200, 200) (testLen, 1)
    def nextTest(self):
        if self.now == self.len:
            return None, None
        nowLen = min(predictBatch, self.len - self.now)
        testData = np.zeros((predictBatch, 3, imageH, imageW))
        ct = 0
        for i in self.idSet[self.now:self.now + nowLen]:
            testData[ct] = readImageInitial(join(testImage, str(i) + '.jpg'))
            ct += 1
        res = testData, self.idSet[self.now:self.now + nowLen]
        self.now += nowLen
        return res
