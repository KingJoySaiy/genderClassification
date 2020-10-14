import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable

from constant.dataset import MyDataSet


# torch.manual_seed(1)    # reproducible


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 200, 200)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 200, 200)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 100, 100)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 100, 100)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 100, 100)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 50, 50)
        )
        self.out = nn.Linear(32 * 50 * 50, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate

train_data = MyDataSet(
    isTrain=True,  # this is training data
    trainSize=700,
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = MyDataSet(
    isTrain=False,
    trainSize=700,
    transform=torchvision.transforms.ToTensor(),
)

test_x = test_data.testData / 255
test_y = test_data.testLabel

test_x = Variable(test_x).double()
test_y = Variable(test_y).type(torch.LongTensor)

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        # print(type(b_x), type(b_y))
        b_x = torch.tensor(b_x, dtype=torch.float32)
        output = cnn(b_x)[0]  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 10 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('epoch: ', epoch, '| step: ', step, '| train loss: %.4f' % loss.data.numpy(),
                  '| test accuracy: %.2f' % accuracy)
