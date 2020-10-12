# import torch
import readFile

x, y = readFile.getTrainData()
z = readFile.getTestData()

print(type(x), type(y), type(z))
# print(y)