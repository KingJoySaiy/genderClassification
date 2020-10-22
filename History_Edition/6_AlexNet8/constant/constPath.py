'''
# for windows laptop
rootPath = 'jiangnan2020_Simple\\'
trainImage = rootPath + 'train\\train\\'
testImage = rootPath + 'test\\test\\'
modelPath = 'model\\AlexNet8.pkl'
trainBatch = 50
predictBatch = 50
'''

rootPath = 'jiangnan2020/'
trainImage = rootPath + 'train/train/'
testImage = rootPath + 'test/test/'
modelPath = 'model/AlexNet8.pkl'
trainBatch = 900
predictBatch = 500


trainCSV = rootPath + 'train.csv'
testCSV = rootPath + 'test.csv'
submitCSV = rootPath + 'submit.csv'

imageW = 227
imageH = 227
globalSeed = 233

# for training
trainEpochs = 600
oneTotal = 18000 / trainBatch
needTrain = True
needPredict = True
newModel = False

trainProportion = 0.7
learningRate = 0.00001
initialMomentum = 0.9
weightDecay = 1e-4
