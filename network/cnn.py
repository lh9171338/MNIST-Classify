from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes, conv_dims, fc_dims):
        super().__init__()
        assert len(conv_dims) > 0, 'conv_dims can not be empty'
        assert len(fc_dims) > 0, 'fc_dims can not be empty'

        convs, fcs = [], []
        for i in range(len(conv_dims)):
            in_dims = 1 if i == 0 else conv_dims[i - 1]
            convs.append(
                nn.Sequential(
                    nn.Conv2d(in_dims, conv_dims[i], 5),
                    nn.BatchNorm2d(conv_dims[i]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2)
                )
            )

        for i in range(len(fc_dims) - 1):
            fcs.append(
                nn.Sequential(
                    nn.Linear(fc_dims[i], fc_dims[i + 1]),
                    nn.BatchNorm1d(fc_dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )
        fcs.append(nn.Linear(fc_dims[-1], num_classes))

        self.conv = nn.Sequential(*convs)
        self.fc = nn.Sequential(*fcs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

