import os
import torch
from yacs.config import CfgNode
from thop import profile
from network.build import build_model


if __name__ == "__main__":
    os.makedirs('figure', exist_ok=True)
    device = torch.device(f'cuda:{0}')

    # parameter
    name_list = ['RNN', 'LSTM', 'GRU', 'MLP', 'CNN', 'MHSA', 'MHSA-PE', 'MTL', 'MTL-Li', 'CMTL', 'STL', 'Transformer', 'Transformer-PE']

    # model
    memory_list, flops_list, params_list = [], [], []
    for name in name_list:
        arch = name.split('-')[0]
        yaml_file = os.path.join('config', arch + '.yaml')
        cfg = CfgNode.load_cfg(open(yaml_file))
        cfg.arch = arch
        if name == 'MHSA':
            cfg.MHSA.pos_encoding = False
        elif name == 'MHSA-PE':
            cfg.MHSA.pos_encoding = True
        elif name == 'Transformer':
            cfg.Transformer.pos_encoding = False
        elif name == 'Transformer-PE':
            cfg.Transformer.pos_encoding = True
        elif name == 'MTL':
            cfg.MTL.linear = False
        elif name == 'MTL-Li':
            cfg.MTL.linear = True
        model = build_model(cfg).to(device)
        image = torch.randn(1, 1, 28, 28).to(device)
        try:
            model.eval()
            output = model(image)
            memory = (torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)) / 1024 ** 2
            flops, params = profile(model.cpu(), inputs=(image.cpu(),))
            del model, image, output
            torch.cuda.empty_cache()

            flops /= 0.5 * 1024 ** 2
            params /= 1024 ** 1
            memory = round(memory, 1)
            flops = round(flops, 1)
            params = round(params, 1)
            print(memory, flops, params)
            memory_list.append(memory)
            flops_list.append(flops)
            params_list.append(params)
        except:
            print(name)
            memory_list.append(None)
            flops_list.append(None)
            params_list.append(None)

    print(name_list)
    print(memory_list)
    print(flops_list)
    print(params_list)

