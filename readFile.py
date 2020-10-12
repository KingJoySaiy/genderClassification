import csv
import numpy as np
from matplotlib import image

rootPath = 'jiangnan2020_Simple\\'

# Train: [1.jpg, 18000.jpg], csv: {id -> label}
trainCSV = rootPath + 'train.csv'
trainImage = rootPath + 'train\\train\\'

# test: [18001.jpg, 23708.jpg], csv: {id}
# testCSV = rootPath + 'test.csv'
# testImage = rootPath + 'test\\test\\'

imageW = 200
imageH = 200

# # read image numpy.ndarray(200, 200, 3)
# def readImage(path):
#     im = image.imread(path)
#     # print(type(im))
#     return im
#
#
# # training data(trainSize, 200, 200, 3), label:
# def getTrainData():
#     # get training id & label
#     id = []
#     label = []
#     reader = csv.reader(open(trainCSV, 'r'))
#     next(reader)
#     for row in reader:
#         id.append(int(row[0]))
#         label.append(int(row[1]))
#
#     # get matrix of training image
#     data = np.zeros((len(id), imageH, imageW, 3))
#     ct = 0
#     for i in id:
#         data[ct] = readImage(trainImage + str(i) + '.jpg')
#         ct += 1
#     return data, np.array(label).reshape(len(label), 1)
#
#
# # testing data(testSize, 200, 200, 3)
# def getTestData():
#     # get testing id
#     testId = []
#     reader = csv.reader(open(testCSV, 'r'))
#     next(reader)
#     for row in reader:
#         testId.append(int(row[0]))
#
#     # get matrix of testing image
#     data = np.zeros((len(testId), imageH, imageW, 3))
#     ct = 0
#     for i in testId:
#         data[ct] = readImage(testImage + str(i) + '.jpg')
#         ct += 1
#     return data

SIZE = imageW * imageH * 3
ct = 0


def readImage(path):
    global ct
    print(ct)
    ct += 1
    im = image.imread(path).reshape(1, SIZE)
    return im


def getTrainData():
    id = []
    label = []
    reader = csv.reader(open(trainCSV, 'r'))
    next(reader)
    for row in reader:
        id.append(int(row[0]))
        label.append(int(row[1]))

    data = np.zeros((len(id), SIZE))
    ct = 0
    for i in id:
        data[ct] = readImage(trainImage + str(i) + '.jpg')
        ct += 1
    return np.hstack((data, np.array(label).reshape(len(label), 1)))
