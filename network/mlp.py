from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dims):
        super().__init__()
        assert len(hidden_dims) > 0, 'hidden_dims can not be empty'

        fcs = []
        for i in range(len(hidden_dims)):
            in_dim = in_dim if i == 0 else hidden_dims[i - 1]
            fcs.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(inplace=True)
                )
            )
        fcs.append(nn.Linear(hidden_dims[-1], num_classes))

        self.fc = nn.Sequential(*fcs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x