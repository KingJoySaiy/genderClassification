import csv
import numpy as np
from PIL import Image
from constant.constPath import *
import torchvision


def readImage(path):
    with Image.open(path) as im:    # (200, 200)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomRotation(degrees=15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        im = transform(im).numpy()  # (227, 227, 3)
        return im.reshape((3, imageH, imageW))  # (3, 224, 224)


def readImageInitial(path):
    with Image.open(path) as im:    # (200, 200)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
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