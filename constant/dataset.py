from torch.utils.data import Dataset
import csv
import numpy as np
from matplotlib import image
from constant.constPath import imageH, imageW, trainCSV, testCSV, trainImage, testImage
import torch

ct = 0


def readImage(path):
    global ct
    print(ct)
    ct += 1
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
