from os.path import join
'''
# for windows laptop
rootPath = 'jiangnan2020_Simple'
trainBatch = 2
predictBatch = 5
'''

# for linux server
rootPath = 'jiangnan2020'
trainBatch = 72
predictBatch = 50


trainImage = join(rootPath, 'train', 'train')
testImage = join(rootPath, 'test', 'test')
trainCSV = join(rootPath, 'train.csv')
testCSV = join(rootPath, 'test.csv')
submitCSV = join(rootPath, 'submit.csv')
modelPath = join('model', 'ResNet50.pkl')


imageW = 224
imageH = 224
globalSeed = 233
imageTotal = 18000


trainEpochs = 5
learningRate = 1e-3
trainProportion = 0.7
initialMomentum = 0.9
weightDecay = 1e-4


needTrain = True
needCuda = True
needPredict = not needTrain
newModel = True



