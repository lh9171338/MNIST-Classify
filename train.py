import os
import tqdm
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from config.cfg import parse
from network.build import build_model


def train(model, loader, cfg, device):
    # Option
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size)
    loss_func = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(cfg.log_path)

    # Train
    step = 1
    best_accuracy = 0.0
    best_state_dict = None
    for epoch in range(1, cfg.num_epochs + 1):
        # Train
        model.train()

        for images, labels in tqdm.tqdm(loader['train'], desc='train: '):
            step_ = step * cfg.train_batch_size
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg.print_freq == 0:
                lr = scheduler.get_last_lr()[0]
                print(f'epoch: {epoch}/{cfg.num_epochs} | loss: {loss.item():6f} | lr: {lr:e}')
                writer.add_scalar('loss', loss, step_)
                writer.add_scalar('lr', lr, step_)
            step += 1

        # Val
        tp = 0

        model.eval()
        for images, labels in tqdm.tqdm(loader['val'], desc='val: '):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.max(outputs, dim=-1)[1]
            tp += (preds == labels).sum().item()

        accuracy = float(tp) / float(len(loader['val'].dataset))
        print(f'epoch: {epoch}/{cfg.num_epochs} | accuracy: {accuracy:.3f}')
        writer.add_scalar('accuracy', accuracy, step_)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
        print(f'best accuracy: {best_accuracy:.3f}')

        scheduler.step()
    writer.close()

    # Save best model
    model_filename = os.path.join(cfg.model_path, cfg.model_name + '.pkl')
    torch.save(best_state_dict, model_filename)


if __name__ == '__main__':
    # Parameter
    cfg = parse()
    os.makedirs(cfg.model_path, exist_ok=True)

    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    # Seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    if use_gpu:
        torch.cuda.manual_seed_all(cfg.seed)

    # Load model
    model = build_model(cfg.model_name).to(device)

    # Load dataset
    train_dataset = datasets.MNIST(root=cfg.dataset_path, train=True, transform=transforms.ToTensor(), download=cfg.download)
    val_dataset = datasets.MNIST(root=cfg.dataset_path, train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size,
                                               num_workers=cfg.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.test_batch_size,
                                               num_workers=cfg.num_workers, shuffle=True)
    loader = {'train': train_loader, 'val': val_loader}

    # Train network
    train(model, loader, cfg, device)