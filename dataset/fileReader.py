import csv
import numpy as np
from PIL import Image
from constant.constPath import *
import torchvision


class fileReader:
    @staticmethod
    def readImage(path):
        with Image.open(path) as im:  # (200, 200)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomCrop(imageH),
                torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomRotation(degrees=10),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            im = transform(im).numpy()  # (224, 224, 3)
            return im.reshape((3, imageH, imageW))  # (3, 224, 224)

    @staticmethod
    def readImageInitial(path):
        with Image.open(path) as im:  # (200, 200)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(imageH),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            im = transform(im).numpy()  # (224, 224, 3)
            return im.reshape((3, imageH, imageW))  # (3, 224, 224)

    @staticmethod
    def getIdLabelSet():
        idLabel = []
        with open(trainCSV, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                idLabel.append((int(row[0]), int(row[1])))
        return idLabel

    @staticmethod
    def getIdSet():
        testId = []
        with open(testCSV, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                testId.append(int(row[0]))
        return np.array(testId).reshape(len(testId))
