import os
from yacs.config import CfgNode
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument('-s', '--seed', type=int, help='seed')
    parser.add_argument('-a', '--arch', type=str, help='model name')
    parser.add_argument('-m', '--model_name', type=str, help='model name')
    parser.add_argument('--train_batch_size', type=int, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, help='test batch size')
    parser.add_argument('--config_path', type=str, default='config', help='config path')
    parser.add_argument('-c', '--config_file', type=str, help='config filename')

    opts = parser.parse_args()
    if opts.config_file is None:
        opts.config_file = f'{opts.arch}.yaml'
    if opts.model_name is None:
        opts.model_name = opts.arch

    opts_dict = vars(opts)
    opts_list = []
    for key, value in zip(opts_dict.keys(), opts_dict.values()):
        if value is not None:
            opts_list.append(key)
            opts_list.append(value)

    yaml_file = os.path.join(opts.config_path, opts.config_file)
    cfg = CfgNode.load_cfg(open(yaml_file))
    cfg.merge_from_list(opts_list)
    cfg.download = not os.path.exists(os.path.join(cfg.dataset_path, cfg.dataset_name))
    cfg.log_path = f'{cfg.log_path}/{cfg.model_name}'
    cfg.freeze()

    # Print cfg
    # for k, v in cfg.items():
    #     print(f'{k}: {v}')

    return cfg
