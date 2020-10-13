import csv
import numpy as np
from matplotlib import image
from constant.constPath import imageSize, trainCSV, testCSV, trainImage, testImage


def readImage(path):
    im = image.imread(path)  # (200, 200, 3)
    return np.mean(im, axis=2).reshape(imageSize)  # (200 * 200)


# training data:label (trainSize, 200 * 200 + 1)
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
    data = np.zeros((len(id), imageSize))
    ct = 0
    for i in id:
        data[ct] = readImage(trainImage + str(i) + '.jpg')
        ct += 1
    return np.hstack((data, np.array(label).reshape(len(label), 1)))


# predicting data(testSize, 200 * 200 + 1)
def getPredictData():
    # get testing id
    testId = []
    reader = csv.reader(open(testCSV, 'r'))
    next(reader)
    for row in reader:
        testId.append(int(row[0]))

    # get matrix of predicting image
    data = np.zeros((len(testId), imageSize))
    ct = 0
    for i in testId:
        data[ct] = readImage(testImage + str(i) + '.jpg')
        ct += 1
    return np.hstack((data, np.array(testId).reshape(len(testId), 1)))


