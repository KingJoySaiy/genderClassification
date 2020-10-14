# import torch
# from constant.readFile import getPredictData
# from constant.constPath import modelPath, predictSize
#
#
# def startPredict():
#     data = getPredictData()
#     l = data.shape[1]
#
#     net = torch.load(modelPath)
#     test_in = torch.from_numpy(data[:predictSize, :l - 1]).float()
#     test_out = net(test_in)
#
#     id = data[:predictSize, l - 1]
#     label = test_out.max(-1)[1]
#     # for i in range(predictSize):
#     #     print(int(id[i]), int(lab[i]))
#     return id, label
