from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(84, 10)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x