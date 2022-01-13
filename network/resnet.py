import torch.nn as nn


class ResNetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes=None, stride=1, downsample=None):
        super().__init__()
        planes = planes or inplanes // self.expansion

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out
