from torch import nn


class RNN(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        assert model_name in ['RNN', 'LSTM', 'GRU'], 'Unrecognized model name'
        if model_name == 'RNN':
            net = nn.RNN
        elif model_name == 'LSTM':
            net = nn.LSTM
        else:
            net = nn.GRU
        self.rnn = net(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64, 10)
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