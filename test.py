import os
import tqdm
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from config.cfg import parse
from network.build import build_model


def test(model, loader, cfg, device):
    # Test
    model.eval()

    tp = 0
    for images, labels in tqdm.tqdm(loader, desc='test: '):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        preds = torch.max(outputs, dim=-1)[1]
        tp += (preds == labels).sum().item()

    accuracy = float(tp) / float(len(loader.dataset))
    print(f'accuracy: {accuracy:.3f}')


if __name__ == '__main__':
    # Parameter
    cfg = parse()

    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    # Load model
    model = build_model(cfg.model_name).to(device)
    model_filename = os.path.join(cfg.model_path, cfg.model_name + '.pkl')
    state_dict = torch.load(model_filename, map_location=device)
    model.load_state_dict(state_dict)

    # Load dataset
    dataset = datasets.MNIST(root=cfg.dataset_path, train=False, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.test_batch_size,
                                               num_workers=cfg.num_workers, shuffle=False)

    # Test network
    test(model, loader, cfg, device)