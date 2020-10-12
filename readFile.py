import csv
import numpy as np
from matplotlib import image

rootPath = 'C:\\KingJoySaiy\\workspace\\genderClassification\\jiangnan2020_Too_Simple\\'

# Train: [1.jpg, 18000.jpg], csv: {id -> label}
trainCSV = rootPath + 'train.csv'
trainImage = rootPath + 'train\\train\\'

# test: [18001.jpg, 23708.jpg], csv: {id}
testCSV = rootPath + 'test.csv'
testImage = rootPath + 'test\\test\\'

imageW = 200
imageH = 200


# read image numpy.ndarray(200, 200, 3)
def readImage(path):
    im = image.imread(path)
    # print(type(im))
    return im


# Data, Label: image(200 * 200) & label of each trainImage
def getTrainData():
    # get directory of training label {id -> label}
    idLabel = dict()
    reader = csv.reader(open(trainCSV, 'r'))
    next(reader)
    for row in reader:
        idLabel[int(row[0])] = int(row[1])

    # get list of training image
    data = np.zeros(len(idLabel), 1)
    ct = 0
    for i in idLabel.keys():
        data[ct, 0] = readImage(trainImage + str(i) + '.jpg')
        ct += 1
    return data, list(idLabel.values())


# Data: image(200 * 200) of each testImage
def getTestData():
    # get list of test {Id}
    testId = []
    reader = csv.reader(open(testCSV, 'r'))
    next(reader)
    for row in reader:
        testId.append(int(row[0]))

    # get directory of training image {id -> matrix}
    data = np.zeros(len(testId), 1)
    ct = 0
    for i in testId:
        data[ct, 0] = readImage(testImage + str(i) + '.jpg')
        ct += 1
    return data
