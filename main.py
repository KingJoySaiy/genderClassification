import random
import constant.dataset

from train import startTrain
from constant.constPath import submitCSV
import csv

# def writePredictData():
#     id, label = startPredict()
#     writer = csv.writer(open(submitCSV, "w", newline=""))
#
#     writer.writerow(['id', 'label'])
#     for row in range(len(id)):
#         writer.writerow([int(id[row]), int(label[row])])

startTrain()
# writePredictData()
