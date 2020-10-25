import torch
import torch.nn as nn

imageSize = 200 * 200


# 1st Edition: Linear Regression (up to 78.661%)

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(imageSize, 2)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out
