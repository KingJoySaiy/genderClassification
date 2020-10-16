from constant.dataset import *
from predict import startPredict
from train import startTrain
from constant.constPath import *
import csv

setSeed(globalSeed)
startTrain()
startPredict()