import torch.nn as nn


# 4th Edition: AlexNet-8 (up to 88.507%)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0),
            nn.Conv2d(96, 192, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0),
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
