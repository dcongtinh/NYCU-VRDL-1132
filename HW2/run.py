import os
import json
import time
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models import *
from util import *
from dataset import DigitDataset, get_test_img
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from copy import deepcopy
from torch.optim import lr_scheduler
from torch import optim

parser = argparse.ArgumentParser()

parser.add_argument('-e', "--epochs", type=int, default=12)
parser.add_argument('-l', "--lr", type=float, default=0.0001)
parser.add_argument('-m', "--backbone", type=str, default="CSPWithFPN",
                    choices=['CSPWithFPN', 'Resnet50', 'efficientnetv2_s_fpn', 'efficientnet_b0'])

parser.add_argument('-b', "--batch_size", type=int, default=4)
parser.add_argument('-s', "--scheduler", type=str, default='multi_step',
                    choices=['multi_step', 'cosine', 'cosine_warmup', 'reduce', 'constant'])
parser.add_argument('-o', "--optimizer", type=str, default='adam',
                    choices=['adam', 'adaw', 'adagrad', 'sgd'])
parser.add_argument("--clip", type=float, default=0)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--notes", type=str, default='anchor_default')

args = parser.parse_args()
print(args)

seed = 0
print_freq = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


same_seeds(seed)
train_dataset = DigitDataset(
    root="nycu-hw2-data", mode="train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=6)
valid_dataset = DigitDataset(
    root="nycu-hw2-data", mode="valid")
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               collate_fn=valid_dataset.collate_fn, num_workers=6)


FasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2

if args.notes == 'anchor_default':
    # run default
    print('anchor_default')
    min_size, max_size = 800, 1333
    model = FasterRCNN(weights='DEFAULT',
                       weights_backbone='IMAGENET1K_V2',
                       trainable_backbone_layers=args.n_layers,)
else:
    # try different anchor settings
    # reducing the image size helps training speed faster but poor performace and vice versa.
    print('custom_anchor')
    # anchor_sizes = ((32,), (64,), (96,), (128,), (160,))
    # aspect_ratios = ((0.33, 0.5, 0.67),) * len(anchor_sizes)
    # anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # min_size, max_size = 600, 1024
    min_size, max_size = 400, 600

    anchor_generator = AnchorGenerator(
        sizes=(anchor_sizes := ((32,), (64,), (96,), (128,), (160,))),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )

    model = FasterRCNN(weights='DEFAULT',
                       weights_backbone='IMAGENET1K_V2',
                       rpn_anchor_generator=anchor_generator,
                       trainable_backbone_layers=args.n_layers,
                       min_size=min_size,
                       max_size=max_size)

num_classes = 11
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

if args.backbone == 'CSPWithFPN':
    backbone = CSPWithFPN()
    model.backbone = backbone
# elif 'efficientnet' in args.backbone:
#     model.backbone = EfficientNetBackbone(args.backbone)

print('Number of parameters:', sum(p.numel() for p in model.parameters()))
print('Number of trainable parameters:', sum(p.numel()
      for p in model.parameters() if p.requires_grad))

model = model.to(device)


def select_optimizer(model, args):
    momentum, weight_decay = 0.9, 5e-4
    params = [p for p in model.parameters() if p.requires_grad]
    optimizers = {
        'adam': optim.Adam(params, lr=args.lr, weight_decay=3e-4),
        'adamw': optim.AdamW(params, lr=args.lr, eps=1e-5, weight_decay=3e-4),
        'adagrad': optim.Adagrad(params, lr=args.lr, eps=1e-5, weight_decay=3e-4),
        'sgd': optim.SGD(params, lr=args.lr, momentum=momentum, weight_decay=weight_decay),
    }
    return optimizers[args.optimizer]


optimizer = select_optimizer(model, args)

# training
best_val_map, best_epoch = -1, -1
metrics = {
    'train': {
        'loss_classifier': [],
        'loss_box_reg': [],
        'loss_objectness': [],
        'loss_rpn_box_reg': [],
        'loss_total': []
    },
    'val': {'mAP': []},
    'lr': [args.lr]
}


def select_scheduler(optimizer, args):
    # # milestones = [3, 6, 9, 12, 15, 18, 21] #[10, 13, 18]
    # milestones = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23] #[10, 13, 18]
    # gamma = 0.75 #0.1

    # milestones = [10, 13, 18]
    milestones = [6]
    gamma = 0.1

    schedulers = {
        'cosine': lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs),
        'cosine_warmup': lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, args.epochs),
        'reduce': lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.85),
        'multi_step': lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma),
        'constant': None
    }
    return schedulers[args.scheduler]


scheduler = select_scheduler(optimizer, args)
best_model = None

timestamp = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"
result_dir = f"results/{args.backbone}_{args.optimizer}_{args.scheduler}_epoch{args.epochs}_lr{args.lr}_bs{args.batch_size}_mn{min_size}_mx{max_size}_nlayers{args.n_layers}_clip{args.clip}_{args.notes}/{timestamp}"
os.makedirs(result_dir, exist_ok=True)


def combine_loss(loss_dict):
    loss_classifier = loss_dict['loss_classifier']
    loss_box_reg = loss_dict['loss_box_reg']
    loss_objectness = loss_dict['loss_objectness']
    loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']

    return loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
    # return 2 * loss_classifier + 1 * loss_box_reg + 0.5 * loss_objectness + 0.5 * loss_rpn_box_reg


try:
    for epoch in tqdm(range(1, args.epochs+1), desc=f"{args}"):
        model.train()
        start = time.time()
        loss_value = 0
        epoch_loss = deepcopy(metrics['train'])

        for i, (images, targets) in (pbar := tqdm(enumerate(train_dataloader), total=len(train_dataloader))):
            images = list(image.to(device) for image in images)
            targets = [{'boxes': t['boxes'].to(
                device), 'labels': t['labels'].to(device)} for t in targets]

            loss_dict = model(images, targets)
            losses = combine_loss(loss_dict)
            loss_dict['loss_total'] = losses.detach()

            for loss_metric in epoch_loss.keys():
                epoch_loss[loss_metric].append(loss_dict[loss_metric].item())

            loss_value += losses.item()

            optimizer.zero_grad()
            losses.backward()
            if args.clip:
                nn.utils.clip_grad_value_(model.parameters(), args.clip)

            optimizer.step()
            # Print status
            if i % print_freq == 0:
                end = time.time()
                pbar.set_description(
                    f"Epoch: {epoch} | {(i+1)*args.batch_size}/{len(train_dataset)} | {loss_value/print_freq:.6f}")
                start = time.time()
                loss_value = 0

        for loss_metric in epoch_loss.keys():
            metrics['train'][loss_metric].append(
                float(np.mean(epoch_loss[loss_metric])))
        if args.scheduler != 'reduce':
            scheduler.step()
        model.eval()
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        with torch.no_grad():
            val_start = time.time()

            for i, (images, targets) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
                images = list(image.to(device) for image in images)

                output = model(images)

                # Store this batch's results for mAP calculation
                boxes = [t['boxes'].to(device) for t in targets]
                labels = [t['labels'].to(device) for t in targets]

                det_boxes.extend([o['boxes'] for o in output])
                det_labels.extend([o['labels'] for o in output])
                det_scores.extend([o['scores'] for o in output])
                true_boxes.extend(boxes)
                true_labels.extend(labels)

            # Calculate mAP
            _mAP = mAP(det_boxes, det_labels, det_scores,
                       true_boxes, true_labels, device)
            val_end = time.time()

            metrics['val']['mAP'].append(_mAP)
            if _mAP > best_val_map:
                best_val_map = _mAP
                best_epoch = epoch
                best_model = deepcopy(model)

            print(
                f"valid mAP: {_mAP:.4f} | best: {best_val_map:.4f} ({best_epoch}) | epoch_loss: {metrics['train']['loss_total'][-1]:.6f} | lr = {metrics['lr'][-1]:.6f}, time: {val_end - val_start:.3f}")

        if args.scheduler == 'reduce':
            scheduler.step(_mAP)
        metrics['lr'].append(scheduler.get_last_lr()[0])
except:
    print(f'Manually stopped! Best Val mAP: {best_val_map:.6f}')
    if best_model is None:
        os.system(f"rm -rf {result_dir}/")
    print('Passing to test phase for Codabench submission...')

assert best_model is not None, "Best model should be not None."

# testing phase

# best_model = torch.load(
#     "fasterRcnn_CSPWithFPN_mAP0.404874_acc0.816345.pth", weights_only=False).to(device)
# result_dir = ""
# timestamp = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"
# best_val_map = 0.4

best_model.eval()
task1, task2 = [], []
conf_thresh = 0.5  # 0.7681818181818182

for img_id in tqdm(range(1, 13068+1), desc="Testing for Codabench submission"):
    img = get_test_img(img_id).unsqueeze(0).to(device)
    output = best_model(img)[0]

    boxes = output["boxes"].to('cpu').detach().numpy()
    labels = output["labels"].to('cpu').detach().numpy()
    scores = output["scores"].to('cpu').detach().numpy()
    keep_indices = scores > 0
    boxes, labels, scores = boxes[keep_indices], labels[keep_indices], scores[keep_indices]
    boxes = np.around(boxes).astype(float)

    image_boxes, image_digits = [], []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        w, h = abs(x1 - x2), abs(y1 - y2)
        bbox = [min(x1, x2), min(y1, y2), w, h]

        if labels[i] > 0:
            image_boxes.append({
                "image_id": img_id,
                "bbox": bbox,
                "score": float(scores[i]),
                "category_id": int(labels[i])
            })
        if scores[i] > conf_thresh and labels[i] > 0:
            image_digits.append((bbox, labels[i] - 1))

    image_digits.sort(key=lambda x: x[0][0])

    digit_str = ''.join(str(d)
                        for _, d in image_digits) if len(image_digits) else -1

    task1.extend(image_boxes)
    task2.append([img_id, digit_str])

# save submission
model_path = os.path.join(
    result_dir, f"{args.backbone}_mAP{best_val_map:.6f}.pth")

metric_path = os.path.join(result_dir, "metrics.json")
json_path = os.path.join(result_dir, "pred.json")
csv_path = os.path.join(result_dir, "pred.csv")
zip_path = os.path.join(
    result_dir, f"{timestamp}__mAP{best_val_map:.6f}_submission.zip")

torch.save(best_model, model_path)
with open(metric_path, 'w') as f:
    json.dump(metrics, f, indent=4)
with open(json_path, 'w') as f:
    json.dump(task1, f)
pd.DataFrame(task2, columns=['image_id', 'pred_label']).to_csv(
    csv_path, index=False)

os.system(f"zip -j {zip_path} {csv_path} {json_path}")
print(f"Submission files saved: {zip_path}")
