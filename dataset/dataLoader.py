import random
from dataset.readFile import *


class TrainData:
    def __init__(self):
        self.idSet, self.labelSet = getIdLabelSet()
        self.now = 0
        self.trainLen = int(trainBatch * trainProportion)
        self.len = len(self.idSet)

    def shuffle(self):
        newId = list(range(self.len))
        random.shuffle(newId)
        newIdSet = []
        newLabelSet = []
        for i in range(self.len):
            newIdSet.append(self.idSet[newId[i]])
            newLabelSet.append(self.labelSet[newId[i]])
        self.idSet = np.array(newIdSet)
        self.labelSet = np.array(newLabelSet)
        self.now = 0

    # train: (trainLen, 1, 200, 200) (trainLen, 1) valid:(batch - trainLen, 1, 200, 200) (batch - trainLen, 1)
    def nextTrainValid(self):

        trainData = np.zeros((self.trainLen, 3, imageH, imageW))
        validData = np.zeros((trainBatch - self.trainLen, 3, imageH, imageW))

        ct = 0
        for i in self.idSet[self.now:self.now + self.trainLen]:
            trainData[ct] = readImage(join(trainImage, str(i) + '.jpg'))
            ct += 1
        ct = 0
        for i in self.idSet[self.now + self.trainLen:self.now + trainBatch]:
            validData[ct] = readImageInitial(join(trainImage, str(i) + '.jpg'))
            ct += 1

        res = trainData, self.labelSet[self.now:self.now + self.trainLen], validData, self.labelSet[
                                                                                      self.now + self.trainLen:self.now + trainBatch]
        self.now = (self.now + trainBatch) % self.len
        return res


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
