from torch.utils.data import Dataset
import csv
import numpy as np
from matplotlib import image
from constant.constPath import imageH, imageW, trainCSV, testCSV, trainImage, testImage
import torch


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


# predicting data:id (testSize, 1, 200, 200) (testSize, 1)
def getPredictData():
    # get testing id
    testId = []
    reader = csv.reader(open(testCSV, 'r'))
    next(reader)
    for row in reader:
        testId.append(int(row[0]))

    # get matrix of predicting image
    data = np.zeros((len(testId), 1, imageH, imageW))
    ct = 0
    for i in testId:
        data[ct] = readImage(testImage + str(i) + '.jpg')
        ct += 1
    return data, np.array(testId).reshape(len(testId), 1)


class MyDataSet(Dataset):
    def __init__(self, isTrain, trainSize, transform=None):
        self.transform = transform
        self.isTrain = isTrain
        tmpX, tmpY = getTrainData()
        tmpX, tmpY = torch.from_numpy(tmpX[:trainSize]).double(), torch.from_numpy(tmpY[:trainSize]).long()
        self.trainData, self.label = tmpX[:trainSize], tmpY[:trainSize]
        self.testData, self.testLabel = tmpX[trainSize:], tmpY[trainSize:]
        # self.trainData, self.label = torch.from_numpy(tmpX[:trainSize]).double(), torch.from_numpy(
        #     tmpY[:trainSize]).long()
        # self.testData, self.testLabel = torch.from_numpy(tmpX[trainSize:]).double(), torch.from_numpy(
        #     tmpY[trainSize:]).long()

    def __getitem__(self, index):
        if self.isTrain:
            return self.trainData[index], int(self.label[index])
        return self.testData[index], int(self.testLabel[index])

    def __len__(self):
        if self.isTrain:
            return len(self.trainData)
        return len(self.testData)
