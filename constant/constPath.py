# # for windows laptop
rootPath = 'jiangnan2020_Simple\\'
trainImage = rootPath + 'train\\train\\'
testImage = rootPath + 'test\\test\\'
modelPath = 'model\\CNN.pkl'
batchSize = 100

predictSize = 100

# # for linux server
# rootPath = 'jiangnan2020/'
# trainImage = rootPath + 'train/train/'
# testImage = rootPath + 'test/test/'
# modelPath = 'model/CNN.pkl'
# trainSize = 15000
# predictSize = 5708

trainCSV = rootPath + 'train.csv'
testCSV = rootPath + 'test.csv'
submitCSV = rootPath + 'submit.csv'

imageW = 200
imageH = 200
# imageSize = imageW * imageH