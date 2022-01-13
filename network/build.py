import torch
from network.cnn import CNN
from network.mhsa import MHSANet
from network.mtn import MTNet
from network.cmtn import CMTNet
from network.stn import STNet
from network.ntn import NTNet
from network.etn import ETNet
from network.atn import ATNet


def build_model(cfg):
    assert cfg.arch in ['CNN', 'MHSA', 'MTN', 'CMTN', 'STN', 'NTN', 'ETN', 'ATN'], 'Unrecognized model name'
    if cfg.arch == 'CNN':
        model = CNN(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth
        )
    elif cfg.arch == 'MHSA':
        model = MHSANet(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            sr_ratio=cfg.sr_ratio,
            qkv_bias=cfg.qkv_bias,
            pos_encoding=cfg.pos_encoding
        )
    elif cfg.arch == 'MTN':
        model = MTNet(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            sr_ratio=cfg.sr_ratio,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_path=cfg.drop_path,
            linear=cfg.linear
        )
    elif cfg.arch == 'CMTN':
        model = CMTNet(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            sr_ratio=cfg.sr_ratio,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_path=cfg.drop_path,
            pos_encoding=cfg.pos_encoding
        )
    elif cfg.arch == 'STN':
        model = STNet(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_path=cfg.drop_path,
            window_size=cfg.window_size
        )
    elif cfg.arch == 'NTN':
        model = NTNet(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            sr_ratio=cfg.sr_ratio,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_path=cfg.drop_path,
            pos_encoding=cfg.pos_encoding
        )
    elif cfg.arch == 'ETN':
        model = ETNet(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            key_dim=cfg.key_dim,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_path=cfg.drop_path
        )
    elif cfg.arch == 'ATN':
        model = ATNet(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            sr_ratio=cfg.sr_ratio,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_path=cfg.drop_path,
            groups=cfg.groups,
            pos_encoding=cfg.pos_encoding
        )
    else:
        model = None

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
