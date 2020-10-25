import torch.nn as nn


# 2nd Edition: Convolution Neural Network (up to 87.351%)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1, 200, 200)
            nn.Conv2d(1, 16, 5, 1, 2),  # output shape (16, 200, 200)
            nn.ReLU(),  # activation
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # (16, 100, 100)
        )
        self.conv2 = nn.Sequential(  # (16, 100, 100)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32, 100, 100)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 7, 7)
        )
        self.out = nn.Linear(32 * 50 * 50, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)
