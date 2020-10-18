'''
# for windows laptop
rootPath = 'jiangnan2020_Simple\\'
trainImage = rootPath + 'train\\train\\'
testImage = rootPath + 'test\\test\\'
modelPath = 'model\\VGGNet16.pkl'
trainBatch = 2
predictBatch = 5
'''

# for linux server
rootPath = 'jiangnan2020/'
trainImage = rootPath + 'train/train/'
testImage = rootPath + 'test/test/'
modelPath = 'model/VGGNet16.pkl'
trainBatch = 36
predictBatch = 20


trainCSV = rootPath + 'train.csv'
testCSV = rootPath + 'test.csv'
submitCSV = rootPath + 'submit.csv'

imageW = 224
imageH = 224
globalSeed = 233

trainEpochs = 100
oneTotal = 18000 / trainBatch
needTrain = False
needPredict = True
newModel = False

trainPropotion = 0.7
learningRate = 0.00001
initialMomentum = 0.9
weightDecay = 1e-4
