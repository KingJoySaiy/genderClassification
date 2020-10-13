import csv
import numpy as np
from matplotlib import image

# rootPath = 'jiangnan2020_Simple\\'  # for windows
rootPath = 'jiangnan2020/'  # for linux

# Train: [1.jpg, 18000.jpg], csv: {id -> label}
trainCSV = rootPath + 'train.csv'
# trainImage = rootPath + 'train\\train\\'  # for window
trainImage = rootPath + 'train/train/'  # for linux

# test: [18001.jpg, 23708.jpg], csv: {id}
testCSV = rootPath + 'test.csv'
testImage = rootPath + 'test\\test\\'   # for windows
# testImage = rootPath + 'test/test/'   # for linux

imageW = 200
imageH = 200

imageSize = imageW * imageH


def readImage(path):
    im = image.imread(path)  # (200, 200, 3)
    return np.mean(im, axis=2).reshape(imageSize)    # (200, 200)


# training data:label (trainSize, 200*200 + 1)
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


# testing data(testSize, 200*200)
def getTestData():
    # get testing id
    testId = []
    reader = csv.reader(open(testCSV, 'r'))
    next(reader)
    for row in reader:
        testId.append(int(row[0]))

    # get matrix of testing image
    data = np.zeros((len(testId), imageSize))
    ct = 0
    for i in testId:
        data[ct] = readImage(testImage + str(i) + '.jpg')
        ct += 1
    return np.hstack((data, np.array(testId).reshape(len(testId), 1)))