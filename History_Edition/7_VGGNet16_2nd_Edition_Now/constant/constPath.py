from os.path import join
'''
# for windows laptop
rootPath = 'jiangnan2020_Simple'
trainBatch = 2
predictBatch = 5
'''

# for linux server
rootPath = 'jiangnan2020'
trainBatch = 36
predictBatch = 20


trainImage = join(rootPath, 'train', 'train')
testImage = join(rootPath, 'test', 'test')
trainCSV = join(rootPath, 'train.csv')
testCSV = join(rootPath, 'test.csv')
submitCSV = join(rootPath, 'submit.csv')
modelPath = join('model', 'VGGNet16.pkl')


imageW = 224
imageH = 224
globalSeed = 233


trainEpochs = 500
learningRate = 0.000001
trainProportion = 0.7
initialMomentum = 0.9
weightDecay = 1e-4


needCuda = True
needTrain = True
needPredict = False
newModel = False



