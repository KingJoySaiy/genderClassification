# for windows laptop
rootPath = 'jiangnan2020_Simple\\'
trainImage = rootPath + 'train\\train\\'
testImage = rootPath + 'test\\test\\'
modelPath = 'model\\CNN.pkl'
trainBatch = 100
predictBatch = 50

'''
# for linux server
rootPath = 'jiangnan2020/'
trainImage = rootPath + 'train/train/'
testImage = rootPath + 'test/test/'
modelPath = 'model/CNN.pkl'
trainSize = 1000
predictBatch = 500
'''

trainCSV = rootPath + 'train.csv'
testCSV = rootPath + 'test.csv'
submitCSV = rootPath + 'submit.csv'

imageW = 200
imageH = 200
globalSeed = 233

# learningRate = 0.001  # 1-200 epoch
learningRate = 0.0001  # 200-600 epoch
