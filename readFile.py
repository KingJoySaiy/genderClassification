import csv
import numpy as np
from PIL import Image

rootPath = 'D:\\KingJoySaiy\\workspace\\genderClassification\\jiangnan2020_Simple\\'

# Train: [1.jpg, 18000.jpg], csv: {id -> label}
trainCSV = rootPath + 'train.csv'
trainImage = rootPath + 'train\\train\\'

# test: [18001.jpg, 23708.jpg], csv: {id}
testCSV = rootPath + 'test.csv'
testImage = rootPath + 'test\\test\\'

imageW = 200
imageH = 200


def readImage(path):
    img = Image.open(path)
    pix = img.load()
    dataX = np.zeros((imageH, imageW, 1), dtype=np.float)
    for x in range(imageH):
        for y in range(imageW):
            r, g, b = pix[y, x]
            dataX[x, y, 0] = (r + g + b) // 3
    return dataX


# idData, idLabel: image(200 * 200) & label of each trainImage
def getTrainData():
    # get directory of training label {id -> label}
    idLabel = dict()
    reader = csv.reader(open(trainCSV, 'r'))
    next(reader)
    for row in reader:
        idLabel[int(row[0])] = row[1]

    # get directory of training image {id -> matrix}
    idData = dict()
    for i in idLabel.keys():
        idData[i] = readImage(trainImage + str(i) + '.jpg')
    return idData, idLabel


# idData: image(200 * 200) of each testImage
def getTestData():
    # get list of test {Id}
    testId = []
    reader = csv.reader(open(testCSV, 'r'))
    next(reader)
    for row in reader:
        testId.append(int(row[0]))

    # get directory of training image {id -> matrix}
    idData = dict()
    for i in testId:
        idData[i] = readImage(testImage + str(i) + '.jpg')
    return idData
