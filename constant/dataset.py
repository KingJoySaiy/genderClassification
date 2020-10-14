from torch.utils.data import Dataset
import csv
import numpy as np
from matplotlib import image
from constant.constPath import imageH, imageW, trainCSV, testCSV, trainImage, testImage
import torch
import random

# ct = 0


def readImage(path):
    # global ct
    # print(ct)
    # ct += 1
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


class TrainData:
    def __init__(self, batch):
        self.idSet, self.labelSet = getIdLabelSet()
        self.now = 0
        self.batch = batch
        self.trainLen = int(self.batch * 0.7)
        self.len = len(self.idSet)

    def reset(self):
        newId = range(self.len)
        random.shuffle(newId)
        newIdSet = []
        newLabelSet = []
        for i in range(self.len):
            newIdSet[i] = self.idSet[newId[i]]
            newLabelSet[i] = self.labelSet[newId[i]]
        self.idSet = newIdSet
        self.labelSet = newLabelSet
        self.now = 0

    # train(trainLen, 1, 200, 200) (trainLen, 1) valid:(batch - trainLen, 1, 200, 200) (batch - trainLen, 1)
    def nextTrainValid(self):

        trainData = np.zeros((self.trainLen, 1, imageH, imageW))
        validData = np.zeros((self.batch - self.trainLen, 1, imageH, imageW))

        ct = 0
        for i in self.idSet[self.now:self.now + self.trainLen]:
            trainData[ct] = readImage(trainImage + str(self.idSet[i]) + '.jpg')
            ct += 1
        ct = 0
        for i in self.idSet[self.now + self.trainLen:self.now + self.batch]:
            validData[ct] = readImage(trainImage + str(self.idSet[i]) + '.jpg')
            ct += 1

        res = trainData, self.labelSet[self.now:self.now + self.trainLen], validData, self.labelSet[self.now + self.trainLen:self.now + self.batch]
        # self.now = (self.now + self.batch) % self.len
        self.now += self.batch
        if self.now + self.batch > self.len:
            self.now = 0
        return res
