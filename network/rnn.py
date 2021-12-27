from torch import nn


class RNN(nn.Module):
    def __init__(self, arch, in_dim, num_classes, hidden_size=64, num_layers=1):
        super().__init__()
        assert arch in ['RNN', 'LSTM', 'GRU'], 'Unrecognized model name'
        if arch == 'RNN':
            net = nn.RNN
        elif arch == 'LSTM':
            net = nn.LSTM
        else:
            net = nn.GRU
        self.rnn = net(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """

        :param x: [B, L, C]
        :return:
        """
        x = x.squeeze(1)
        r_out, _ = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        out = self.softmax(out)

        return out