import torch.nn as nn
import torch.nn.functional as F
from network.resnet import ResNetBottleneck


class CNN(nn.Module):
    def __init__(self, num_classes, embed_dim, depth):
        super().__init__()

        self.patch_embed = ResNetBottleneck(1, embed_dim // 2)

        layers = [ResNetBottleneck(embed_dim) for _ in range(depth)]
        self.layer = nn.Sequential(*layers)

        self.head = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.patch_embed(x)
        x = self.layer(x)
        x = x.mean(-1).mean(-1)
        x = self.head(x)
        x = self.softmax(x)

        return x

