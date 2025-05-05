import argparse
from detectron2.data import DatasetCatalog
from trainer import Trainer, TrainerAdam, TrainerAdamW, TrainerAdagrad
from utils import *
from detectron2.modeling import build_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str,
                        default='mask_rcnn_R_50_DC5_3x')

    parser.add_argument('-e', '--epochs', type=int, default=225)
    parser.add_argument('-b', '--batch_size', type=int, default=14)
    parser.add_argument('-l', '--base_lr', type=float, default=0.001)

    parser.add_argument('-s', '--scheduler', type=str, default='WarmupCosineLR',
                        choices=['WarmupMultiStepLR', 'WarmupCosineLR'])
    parser.add_argument('-o', '--optimizer', type=str, default='Adagrad',
                        choices=['SGD', 'Adam', 'AdamW', 'Adagrad'])

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='tensorboard')

    args = parser.parse_args()
    print(args)

    cfg = set_cfg(args)
    DatasetCatalog.register("Instance_Segmentation_DATA",
                            lambda: read_json("nycu-hw3-data/train__annot.json"))
    # Build your model
    model = build_model(cfg)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    print('Number of trainable parameters:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    del model

    if args.optimizer == 'SGD':
        trainer = Trainer(cfg)
    elif args.optimizer == 'Adam':
        trainer = TrainerAdam(cfg)
    elif args.optimizer == 'AdamW':
        trainer = TrainerAdamW(cfg)
    elif args.optimizer == 'Adagrad':
        trainer = TrainerAdagrad(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()
