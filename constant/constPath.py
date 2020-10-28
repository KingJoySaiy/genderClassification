from os.path import join
import random
import numpy
import torch

# for windows laptop
rootPath = 'jiangnan2020_Simple'
trainBatch = 3
testBatch = 3
'''
# for linux server
rootPath = 'jiangnan2020'
trainBatch = 60
testBatch = 40
'''

# constant (not need to change)
trainImage = join(rootPath, 'train', 'train')
testImage = join(rootPath, 'test', 'test')
trainCSV = join(rootPath, 'train.csv')
testCSV = join(rootPath, 'test.csv')
submitCSV = join(rootPath, 'submit.csv')
modelPath = join('savedModel', 'ResNet50.pkl')
imageW = 224
imageH = 224
trainSize = 17000
needCuda = False

# need to change during training
randomSeed = 996  # 1 model -> 1 seed
learningRate = 1e-3  # 1e-3 ~ 1e-6
saveModelEpoch = 50  # 100, 50, 30
trainEpochs = 3
newModel = True


def setSeed(seed=randomSeed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
