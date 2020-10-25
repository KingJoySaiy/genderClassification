from os.path import join
'''
# for windows laptop
rootPath = 'jiangnan2020_Simple'
trainBatch = 2
predictBatch = 5
'''

# for linux server
rootPath = 'jiangnan2020'
trainBatch = 30
predictBatch = 20


trainImage = join(rootPath, 'train', 'train')
testImage = join(rootPath, 'test', 'test')
trainCSV = join(rootPath, 'train.csv')
testCSV = join(rootPath, 'test.csv')
submitCSV = join(rootPath, 'submit.csv')
modelPath = join('model', 'ResNet152.pkl')


imageW = 224
imageH = 224
globalSeed = 233
imageTotal = 18000


trainEpochs = 600
learningRate = 1e-6
trainProportion = 0.9
initialMomentum = 0.9
weightDecay = 1e-4


needTrain = False
needCuda = True
needPredict = not needTrain
newModel = False



