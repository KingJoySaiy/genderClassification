'''
# for windows laptop
rootPath = 'jiangnan2020_Simple\\'
trainImage = rootPath + 'train\\train\\'
testImage = rootPath + 'test\\test\\'
modelPath = 'model\\LeNet5.pkl'
trainBatch = 100
predictBatch = 50
'''

# for linux server
rootPath = 'jiangnan2020/'
trainImage = rootPath + 'train/train/'
testImage = rootPath + 'test/test/'
modelPath = 'model/LeNet5.pkl'
trainBatch = 1000
predictBatch = 500


trainCSV = rootPath + 'train.csv'
testCSV = rootPath + 'test.csv'
submitCSV = rootPath + 'submit.csv'

imageW = 200
imageH = 200
globalSeed = 233

trainPropotion = 0.95
learningRate = 1e-4
initialMomentum = 0.9
weightDecay = 1e-4
