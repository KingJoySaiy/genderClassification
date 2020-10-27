from os.path import join
import random
import numpy
import torch

'''
# for windows laptop
rootPath = 'jiangnan2020_Simple'
trainBatch = 5
validBatch = 5
predictBatch = 5
'''
# for linux server
rootPath = 'jiangnan2020'
trainBatch = 60
validBatch = 12
predictBatch = 40


trainImage = join(rootPath, 'train', 'train')
testImage = join(rootPath, 'test', 'test')
trainCSV = join(rootPath, 'train.csv')
testCSV = join(rootPath, 'test.csv')
submitCSV = join(rootPath, 'submit.csv')
modelPath = join('savedModel', 'ResNet50.pkl')


imageW = 224
imageH = 224
randomSeed = 996
imageTotal = 18000


trainEpochs = 600
learningRate = 1e-6
trainProportion = 0.7


needCuda = True
newModel = False


def setSeed(seed=randomSeed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
