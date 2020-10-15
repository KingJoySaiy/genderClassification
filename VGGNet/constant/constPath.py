'''
# for windows laptop
rootPath = 'jiangnan2020_Simple\\'
trainImage = rootPath + 'train\\train\\'
testImage = rootPath + 'test\\test\\'
modelPath = 'model\\VGGNet.pkl'
trainBatch = 20
predictBatch = 10
'''

# for linux server
rootPath = 'jiangnan2020/'
trainImage = rootPath + 'train/train/'
testImage = rootPath + 'test/test/'
modelPath = 'model/VGGNet.pkl'
trainBatch = 30
predictBatch = 20


trainCSV = rootPath + 'train.csv'
testCSV = rootPath + 'test.csv'
submitCSV = rootPath + 'submit.csv'

imageW = 224
imageH = 224
globalSeed = 233

# learningRate = 0.001  # 1-200 epoch
learningRate = 1e-7
initialMomentum = 0.9
weightDecay = 1e-4