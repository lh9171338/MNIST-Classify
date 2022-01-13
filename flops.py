import os
import torch
from yacs.config import CfgNode
from thop import profile
from network.build import build_model


if __name__ == "__main__":
    os.makedirs('figure', exist_ok=True)
    device = torch.device(f'cuda:{0}')

    # parameter
    name_list = ['CNN', 'MHSA', 'NTN', 'MTN', 'MTN-Li', 'CMTN', 'STN', 'ETN', 'ATN']

    # model
    memory_list, flops_list, params_list = [], [], []
    for name in name_list:
        arch = name.split('-')[0]
        yaml_file = os.path.join('config', arch + '.yaml')
        cfg = CfgNode.load_cfg(open(yaml_file))
        cfg.arch = arch
        cfg.pos_encoding = True
        cfg.depth = 1
        cfg.train_batch_size = 32
        if name == 'MTN':
            cfg.linear = False
        elif name == 'MTN-Li':
            cfg.linear = True
        model = build_model(cfg).to(device)
        image = torch.randn(cfg.train_batch_size, 1, 28, 28).to(device)
        try:
            model.eval()
            output = model(image)
            memory = (torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)) / 1024 ** 3
            flops, params = profile(model.cpu(), inputs=(image.cpu(),), verbose=False)
            del model, image, output
            torch.cuda.empty_cache()

            flops /= 0.5 * 1024 ** 3
            params /= 1024 ** 1
            memory = round(memory, 1)
            flops = round(flops, 1)
            params = round(params, 1)
            print(memory, flops, params)
            memory_list.append(memory)
            flops_list.append(flops)
            params_list.append(params)
        except RuntimeError:
            print(name)
            memory_list.append(None)
            flops_list.append(None)
            params_list.append(None)

    print(name_list)
    print(memory_list)
    print(flops_list)
    print(params_list)

