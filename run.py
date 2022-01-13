import os
from yacs.config import CfgNode
from test import run


def parse(name, depth):
    arch = name.split('-')[0]
    yaml_file = os.path.join('config', arch + '.yaml')
    cfg = CfgNode.load_cfg(open(yaml_file))
    cfg.arch = arch
    # cfg.train_dataset_ratio = 0.1
    cfg.pos_encoding = True
    cfg.depth = depth
    if cfg.model_name == '':
        cfg.model_name = f'{cfg.train_dataset_ratio}-{cfg.arch}-{cfg.depth}'
    if name == 'MTN':
        cfg.linear = False
    elif name == 'MTN-Li':
        cfg.linear = True
        cfg.model_name = f'{cfg.model_name}-Li'
    cfg.download = not os.path.exists(os.path.join(cfg.dataset_path, cfg.dataset_name))
    cfg.log_path = f'{cfg.log_path}/{cfg.model_name}'
    # cfg.print_freq = 10

    return cfg


if __name__ == "__main__":
    # parameter
    name_list = ['CNN', 'MHSA', 'NTN', 'MTN', 'MTN-Li', 'CMTN', 'STN', 'ETN', 'ATN']

    # model
    for name in name_list:
        for depth in range(1, 2):
            cfg = parse(name, depth)
            while True:
                try:
                    run(cfg)
                except:
                    print(f'Error: batch size: {cfg.train_batch_size}')
                    cfg.train_batch_size //= 2
                    cfg.test_batch_size //= 2
                    if cfg.train_batch_size < 1:
                        exit(-1)
                else:
                    break
