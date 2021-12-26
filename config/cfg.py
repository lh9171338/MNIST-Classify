import os
from yacs.config import CfgNode
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-m', '--model_name', type=str, default='RNN', choices=['RNN', 'LSTM', 'GRU', 'MLP', 'CNN'],
                        help='model name')
    parser.add_argument('--train_batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
    parser.add_argument('--config_path', type=str, default='config', help='config path')
    parser.add_argument('-c', '--config_file', type=str, help='config filename')

    opts = parser.parse_args()
    if opts.config_file is None:
        opts.config_file = f'{opts.model_name}.yaml'

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
    for k, v in cfg.items():
        print(f'{k}: {v}')

    return cfg
