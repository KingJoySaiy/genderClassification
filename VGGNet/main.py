import numpy

from constant.dataset import *
from predict import startPredict
from train import startTrain
from constant.constPath import *
import csv

setSeed(globalSeed)
startTrain()
startPredict()

# xx = numpy.zeros((200, 200, 3))
# print(type(xx), xx.shape)
# xx.resize((3, 224, 224))
# print(xx.shape)
