import torch.nn as nn


# 5th Edition: VggNet-16 (up to 90.189%)

def Conv3x3BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(True)
    )


class VGGNet(nn.Module):
    def __init__(self):  # (trainSize, 3, 224, 224)
        super(VGGNet, self).__init__()
        block_nums = [2, 2, 3, 3, 3]  # vgg16
        # block_nums = [2, 2, 4, 4, 4]  # vgg19
        self.stage1 = self._make_layers(3, 64, block_nums[0])
        self.stage2 = self._make_layers(64, 128, block_nums[1])
        self.stage3 = self._make_layers(128, 256, block_nums[2])
        self.stage4 = self._make_layers(256, 512, block_nums[3])
        self.stage5 = self._make_layers(512, 512, block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.Dropout(0.2),
            nn.Linear(4096, 2)
        )

        self._init_params()

    @staticmethod
    def _make_layers(in_channels, out_channels, block_num):
        layers = [Conv3x3BNReLU(in_channels, out_channels)]
        for i in range(1, block_num):
            layers.append(Conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(2, 2, ceil_mode=False))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
