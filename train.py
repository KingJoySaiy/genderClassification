import torch
import torch.nn as nn
from constant.dataset import TrainData
from constant.constPath import *
from torch.nn import functional as F

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 47 * 47, 120) # ����������д����conv,�ٷ��̳��������Բ�
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


def startTrain():
    data = TrainData()

    # data, label = getTrainData()
    # n, t, x, y = data.shape
    print('start initializing!')

    # net = LeNet5()
    net = torch.load(modelPath)
    net.cuda()
    criterion = nn.CrossEntropyLoss()  # ʹ��CrossEntropyLoss��ʧ
    optm = torch.optim.Adam(net.parameters())  # Adam�Ż�
    # optm = torch.optim.SGD(net.parameters(), momentum=initialMomentum, lr=learningRate, weight_decay=weightDecay)
    epochs = 90  # 18 -> total

    print('start training!')
    for i in range(epochs):
        if i % 18 == 0:
            data.shuffle()
        trainData, trainLabel, validData, validLabel = data.nextTrainValid()

        # ָ��ģ��Ϊѵ��ģʽ�������ݶ�
        net.train()
        # ����ֵ����Ҫת����torch��Tensor
        x = torch.from_numpy(trainData).float().cuda()
        y = torch.from_numpy(trainLabel).long().cuda()
        y_hat = net(x)
        # print(type(y), y.shape)
        # print(type(y_hat), y_hat.shape)

        loss = criterion(y_hat, y)  # ������ʧ
        optm.zero_grad()  # ǰһ������ʧ����
        loss.backward()  # ���򴫲�
        optm.step()  # �Ż�

        net.eval()
        test_in = torch.from_numpy(validData).float().cuda()
        test_l = torch.from_numpy(validLabel).long().cuda()
        test_out = net(test_in)
        # ʹ�����ǵĲ��Ժ�������׼ȷ��
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy:{:.2f}".format(i + 1, loss.item(), accu))

    torch.save(net, modelPath)
