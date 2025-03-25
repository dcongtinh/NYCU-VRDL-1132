import argparse
import os
import time
import json
import copy
from datetime import datetime
from tqdm.auto import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import lr_scheduler
from timm.loss import LabelSmoothingCrossEntropy
import timm
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Image Classification Training")

    parser.add_argument("--model", type=str,
                        default="seresnextaa101d_32x8d.sw_in12k_ft_in1k")
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default='adagrad',
                        choices=['adagrad', 'sgd', 'adam', 'adamw'])
    parser.add_argument("--scheduler", type=str, default='cosine',
                        choices=['cosine', 'cosine_warmup', 'reduce', 'constant'])
    parser.add_argument("--loss", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'label_cross_entropy'])
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--pretrained", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    print(args)
    return args


def same_seeds(seed=0):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(model_name, pretrained, num_classes=100):
    try:
        print('Pretrained weights:', pretrained)
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes)
    except:
        print('WITHOUT pretrained weights...')
        model = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes)
    return model


def setup_data_loaders(data_dir, transforms, batch_size, num_workers=6):
    # Prepare DataLoader for train, validation, and test datasets
    train_dataset = datasets.ImageFolder(os.path.join(
        data_dir, "train"), transform=transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(
        data_dir, "val"), transform=transforms['val'])
    test_dataset = datasets.ImageFolder(os.path.join(
        data_dir, "test"), transform=transforms['val'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('Classes', train_dataset.classes)
    test_imgs = [filepath.split('/')[-1].split('.')[0]
                 for filepath, _ in test_dataset.imgs]
    return train_loader, val_loader, test_loader, test_imgs


def select_optimizer(model, args):
    optimizers = {
        'adam': optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4),
        'adamw': optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5, weight_decay=3e-4),
        'adagrad': optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-5, weight_decay=3e-4),
        'sgd': optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4),
    }
    return optimizers[args.optimizer]


def select_loss(args):
    losses = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'label_cross_entropy': LabelSmoothingCrossEntropy(),
    }
    return losses[args.loss]


def select_scheduler(optimizer, args):
    schedulers = {
        'cosine': lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch),
        'cosine_warmup': lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 0, args.n_epoch),
        'reduce': lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.9),
        'constant': None
    }
    return schedulers[args.scheduler]


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, args, result_dir, device):
    # Train and evaluate the model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    results = {'train': {'acc': [], 'loss': []},
               'val': {'acc': [], 'loss': []},
               'lr': []}

    start_time = time.time()

    for epoch in range(args.n_epoch):
        print(f'Epoch {epoch}/{args.n_epoch - 1}\n' + '-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Set model to {phase} mode
            model.train() if phase == 'train' else model.eval()

            running_loss, running_corrects = 0.0, 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if args.clip:
                            nn.utils.clip_grad_value_(
                                model.parameters(), args.clip)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                break

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            results[phase]['loss'].append(epoch_loss)
            results[phase]['acc'].append(epoch_acc)

            # Update scheduler and save learning rate
            if phase == 'val' and scheduler and args.scheduler == 'reduce':
                scheduler.step(epoch_acc)
                results['lr'].append(scheduler.get_last_lr()[0])
                print(
                    f"bs: {args.batch_size}, learning rate: {results['lr'][-1]:.8f}")
            elif phase == 'train' and scheduler and args.scheduler != 'reduce':
                scheduler.step()
                results['lr'].append(scheduler.get_last_lr()[0])
                print(
                    f"bs: {args.batch_size}, learning rate: {results['lr'][-1]:.8f}")

            # Update best model (deep copy the model)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print(
                f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Best Acc: {best_acc:.4f}')

        print()

    elapsed_time = time.time() - start_time
    print(
        f'Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, f"{result_dir}/model.pth")
    return model, results


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    same_seeds(args.seed)

    model = create_model(args.model, args.pretrained).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    model_config = timm.create_model(args.model, pretrained=False)
    data_config = timm.data.resolve_model_data_config(model_config)
    del model_config
    transforms = {
        'train': timm.data.create_transform(**data_config, is_training=True),
        'val': timm.data.create_transform(**data_config, is_training=False),
    }

    train_loader, val_loader, test_loader, test_imgs = setup_data_loaders(
        "./data", transforms, args.batch_size)

    dataloaders = {'train': train_loader,
                   'val': val_loader, 'test': test_loader}
    dataset_sizes = {k: len(v.dataset) for k, v in dataloaders.items()}

    optimizer = select_optimizer(model, args)
    criterion = select_loss(args)
    scheduler = select_scheduler(optimizer, args)

    result_dir = f"results/{args.model}/{datetime.now().strftime('%Y%m%d__%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    model, results = train_model(model, criterion, optimizer, scheduler,
                                 dataloaders, dataset_sizes, args, result_dir, device)
    results['args'] = vars(args)

    with open(f"{result_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    predictions = []
    model.eval()
    # no autograd makes validation go faster
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())

    submission = pd.DataFrame(
        {'image_name': test_imgs, 'pred_label': predictions})
    submission.to_csv(f"{result_dir}/prediction.csv", index=False)
