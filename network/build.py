import torch
from network.rnn import RNN
from network.cnn import CNN
from network.mlp import MLP
from network.mhsa import MHSANet
from network.mtl import MTLNet
from network.cmtl import CMTLNet
from network.stl import STLNet
from network.transformer import Transformer


def build_model(cfg):
    assert cfg.arch in ['RNN', 'LSTM', 'GRU', 'MLP', 'CNN', 'MHSA', 'MTL', 'CMTL', 'STL', 'Transformer'], 'Unrecognized model name'
    if cfg.arch == 'MLP':
        model = MLP(
            in_dim=cfg.MLP.in_dim,
            num_classes=cfg.MLP.num_classes,
            hidden_dims=cfg.MLP.hidden_dims
        )
    elif cfg.arch == 'CNN':
        model = CNN(
            num_classes=cfg.CNN.num_classes,
            conv_dims=cfg.CNN.conv_dims,
            fc_dims=cfg.CNN.fc_dims
        )
    elif cfg.arch == 'MHSA':
        model = MHSANet(
            in_dim=cfg.MHSA.in_dim,
            num_classes=cfg.MHSA.num_classes,
            embed_dim=cfg.MHSA.embed_dim,
            depth=cfg.MHSA.depth,
            num_heads=cfg.MHSA.num_heads,
            sr_ratio=cfg.MHSA.sr_ratio,
            qkv_bias=cfg.MHSA.qkv_bias,
            width=cfg.MHSA.width,
            height=cfg.MHSA.height,
            pos_encoding=cfg.MHSA.pos_encoding
        )
    elif cfg.arch == 'MTL':
        model = MTLNet(
            in_dim=cfg.MTL.in_dim,
            num_classes=cfg.MTL.num_classes,
            embed_dim=cfg.MTL.embed_dim,
            depth=cfg.MTL.depth,
            num_heads=cfg.MTL.num_heads,
            sr_ratio=cfg.MTL.sr_ratio,
            mlp_ratio=cfg.MTL.mlp_ratio,
            qkv_bias=cfg.MTL.qkv_bias,
            linear=cfg.MTL.linear
        )
    elif cfg.arch == 'CMTL':
        model = CMTLNet(
            in_dim=cfg.CMTL.in_dim,
            num_classes=cfg.CMTL.num_classes,
            embed_dim=cfg.CMTL.embed_dim,
            depth=cfg.CMTL.depth,
            num_heads=cfg.CMTL.num_heads,
            sr_ratio=cfg.CMTL.sr_ratio,
            mlp_ratio=cfg.CMTL.mlp_ratio,
            qkv_bias=cfg.CMTL.qkv_bias
        )
    elif cfg.arch == 'STL':
        model = STLNet(
            in_dim=cfg.STL.in_dim,
            num_classes=cfg.STL.num_classes,
            input_size=cfg.STL.input_size,
            window_size=cfg.STL.window_size,
            embed_dim=cfg.STL.embed_dim,
            depth=cfg.STL.depth,
            num_heads=cfg.STL.num_heads,
            mlp_ratio=cfg.STL.mlp_ratio,
            qkv_bias=cfg.STL.qkv_bias
        )
    elif cfg.arch == 'Transformer':
        model = Transformer(
            in_dim=cfg.Transformer.in_dim,
            num_classes=cfg.Transformer.num_classes,
            embed_dim=cfg.Transformer.embed_dim,
            depth=cfg.Transformer.depth,
            num_heads=cfg.Transformer.num_heads,
            sr_ratio=cfg.Transformer.sr_ratio,
            mlp_ratio=cfg.Transformer.mlp_ratio,
            qkv_bias=cfg.Transformer.qkv_bias,
            width=cfg.Transformer.width,
            height=cfg.Transformer.height,
            pos_encoding=cfg.Transformer.pos_encoding
        )
    else:
        model = RNN(
            arch=cfg.arch,
            in_dim=cfg.RNN.in_dim,
            num_classes=cfg.RNN.num_classes,
            hidden_size=cfg.RNN.hidden_size,
            num_layers=cfg.RNN.num_layers
        )

    return model


def build_optimizer(cfg, model):
    assert cfg.optim in ['SGD', 'Adam', 'AdamW'], 'Unrecognized optimizer name'
    if cfg.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)

    return optimizer


def build_scheduler(cfg, optimizer):
    assert cfg.scheduler in ['StepLR', 'Cosine'], 'Unrecognized scheduler name'
    if cfg.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    return scheduler
