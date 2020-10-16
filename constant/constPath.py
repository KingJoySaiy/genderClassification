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

trainPropotion = 0.7
learningRate = 0.00001
initialMomentum = 0.9
weightDecay = 1e-4
