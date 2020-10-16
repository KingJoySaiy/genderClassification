from torch.utils.data import Dataset
import csv
import numpy as np
from matplotlib import image
from constant.constPath import *
import torch
import random


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def readImage(path):
    im = image.imread(path)  # (200, 200, 3)
    return np.mean(im, axis=2).reshape((1, imageH, imageW))  # (1, 200, 200)


# training data:label (trainSize, 1, 200, 200) (trainSize, 1)
def getTrainData():
    # get training id & label
    id = []
    label = []
    reader = csv.reader(open(trainCSV, 'r'))
    next(reader)
    for row in reader:
        id.append(int(row[0]))
        label.append(int(row[1]))

    # get matrix of training image
    data = np.zeros((len(id), 1, imageH, imageW))
    ct = 0
    for i in id:
        data[ct] = readImage(trainImage + str(i) + '.jpg')
        ct += 1
    return data, np.array(label).reshape(len(label), 1)


def getIdLabelSet():
    id = []
    label = []
    reader = csv.reader(open(trainCSV, 'r'))
    next(reader)
    for row in reader:
        id.append(int(row[0]))
        label.append(int(row[1]))
    return np.array(id).reshape(len(id)), np.array(label).reshape(len(id))


def getIdSet():
    testId = []
    reader = csv.reader(open(testCSV, 'r'))
    next(reader)
    for row in reader:
        testId.append(int(row[0]))
    return np.array(testId).reshape(len(testId))


class TrainData:
    def __init__(self):
        self.idSet, self.labelSet = getIdLabelSet()
        self.now = 0
        self.trainLen = int(trainBatch * 0.7)
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

        trainData = np.zeros((self.trainLen, 1, imageH, imageW))
        validData = np.zeros((trainBatch - self.trainLen, 1, imageH, imageW))

        ct = 0
        for i in self.idSet[self.now:self.now + self.trainLen]:
            trainData[ct] = readImage(trainImage + str(i) + '.jpg')
            ct += 1
        ct = 0
        for i in self.idSet[self.now + self.trainLen:self.now + trainBatch]:
            validData[ct] = readImage(trainImage + str(i) + '.jpg')
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
        testData = np.zeros((predictBatch, 1, imageH, imageW))
        ct = 0
        for i in self.idSet[self.now:self.now + nowLen]:
            testData[ct] = readImage(testImage + str(i) + '.jpg')
            ct += 1
        res = testData, self.idSet[self.now:self.now + nowLen]
        self.now += nowLen
        return res
