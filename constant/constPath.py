from os.path import join
import random
import numpy
import torch
'''
# for laptop
rootPath = 'jiangnan2020_Simple'
trainBatch = 3
testBatch = 3
'''
# for server
rootPath = 'jiangnan2020'
trainBatch = 70
testBatch = 40


# constant (not need to change)
trainImage = join(rootPath, 'train', 'train')
testImage = join(rootPath, 'test', 'test')
trainCSV = join(rootPath, 'train.csv')
testCSV = join(rootPath, 'test.csv')
submitCSV = join(rootPath, 'submit.csv')
modelPath = join('savedModel', 'ResNet50.pkl')
imageW = 224
imageH = 224
trainSize = 18000
needCuda = True

# need to change during training
randomSeed = 1998  # 1 model -> 1 seed
learningRate = 1e-8  # 1e-3 ~ 1e-8 , each for 1000 epoch (3-5 saveEpoch, 6-8 loss)
saveModelEpoch = 100
trainEpochs = 1028
newModel = False


def setSeed(seed=randomSeed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
