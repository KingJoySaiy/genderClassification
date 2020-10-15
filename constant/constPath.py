# for windows laptop
'''
rootPath = 'jiangnan2020_Simple\\'
trainImage = rootPath + 'train\\train\\'
testImage = rootPath + 'test\\test\\'
modelPath = 'model\\CNN.pkl'
batchSize = 100
'''

# for linux server
rootPath = 'jiangnan2020/'
trainImage = rootPath + 'train/train/'
testImage = rootPath + 'test/test/'
modelPath = 'model/CNN.pkl'
batchSize = 1000

trainCSV = rootPath + 'train.csv'
testCSV = rootPath + 'test.csv'
submitCSV = rootPath + 'submit.csv'

imageW = 200
imageH = 200
globalSeed = 233

# learningRate = 0.001  # 1-200 epoch
learningRate = 0.0001  # 200-600 epoch
# learningRate = 0.00001  # 600-680 epoch
# learningRate = 0.000001  # 680-800 epoch
# learningRate = 1e-7  # 800-900 epoch
# learningRate = 1e-8  # 900- epoch