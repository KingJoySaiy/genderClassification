from readFile import getTrainData, getTestData


class Data:
    def __init__(self):
        """
        self.__trainSize = 18000    # real size
        self.__testSize = 5708
        """
        self.__trainSize = 300  # simple size
        self.__testSize = 100
        self.__trainNow = 0
        self.__testNow = 0
        self.__idData, self.__idLabel = getTrainData()
        self.__testIdData = getTestData()

    def getTrainSize(self):
        return self.__trainSize

    def getTestSize(self):
        return self.__testSize

    def nextTrainBatch(self, batchSize):  # trainSize must be multiple of batchSize
        self.__trainNow += batchSize
        return self.__idData[self.__trainNow - batchSize:self.__trainNow].keys(), \
               self.__idData[self.__trainNow - batchSize:self.__trainNow].values()

    def nextTestBatch(self, batchSize):  # testSize must be multiple of batchSize
        self.__testNow += batchSize
        return self.__testIdData[self.__testNow - batchSize:self.__testNow].keys(), \
               self.__testIdData[self.__testNow - batchSize:self.__testNow].values()
