from network.rnn import RNN
from network.cnn import CNN
from network.mlp import MLP


def build_model(name):
    assert name in ['RNN', 'LSTM', 'GRU', 'MLP', 'CNN'], 'Unrecognized model name'
    if name == 'MLP':
        model = MLP()
    elif name == 'CNN':
        model = CNN()
    else:
        model = RNN(name)

    return model