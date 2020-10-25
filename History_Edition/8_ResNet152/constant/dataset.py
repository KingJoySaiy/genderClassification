import csv
import numpy as np
from PIL import Image
from constant.constPath import *
import torch
import random
import torchvision
import numpy


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def readImage(path):
    with Image.open(path) as im:    # (200, 200)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        im = transform(im).numpy()  # (227, 227, 3)
        return im.reshape((3, imageH, imageW))  # (3, 224, 224)


# training data:label (trainSize, 1, 200, 200) (trainSize, 1)
def getTrainData():
    # get training id & label
    id = []
    label = []
    with open(trainCSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            id.append(int(row[0]))
            label.append(int(row[1]))

    # get matrix of training image
    data = np.zeros((len(id), 3, imageH, imageW))
    ct = 0
    for i in id:
        data[ct] = readImage(join(trainImage, str(i) + '.jpg'))
        ct += 1
    return data, np.array(label).reshape(len(label), 1)


def getIdLabelSet():
    id = []
    label = []
    with open(trainCSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            id.append(int(row[0]))
            label.append(int(row[1]))
    return np.array(id).reshape(len(id)), np.array(label).reshape(len(id))


def getIdSet():
    testId = []
    with open(testCSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            testId.append(int(row[0]))
    return np.array(testId).reshape(len(testId))


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
            validData[ct] = readImage(join(trainImage, str(i) + '.jpg'))
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
            testData[ct] = readImage(join(testImage, str(i) + '.jpg'))
            ct += 1
        res = testData, self.idSet[self.now:self.now + nowLen]
        self.now += nowLen
        return res
