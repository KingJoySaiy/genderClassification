from os.path import join
import random
import numpy
import torch


# constant (not need to change)
rootPath = 'jiangnan2020'
trainImage = join(rootPath, 'train', 'train')
testImage = join(rootPath, 'test', 'test')
trainCSV = join(rootPath, 'train.csv')
testCSV = join(rootPath, 'test.csv')
submitCSV = join(rootPath, 'submit.csv')
modelPath = join('savedModel', 'ResNet50.pkl')
imageW = 224
imageH = 224

'''
# for laptop
trainBatch = 3
testBatch = 3
'''

# for server
trainBatch = 70
testBatch = 40
trainSize = 17500
needCuda = True

# need to change during training
randomSeed = 233
learningRate = 1e-8  # 1e-3 ~ 1e-8 , each for 1000 epoch
trainEpochs = 1000
saveModelEpoch = 25 # save model every 100 epochs
miniLoss = 0.04
newModel = False


def setSeed(seed=randomSeed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
