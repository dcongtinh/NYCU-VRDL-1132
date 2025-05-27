import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR

from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from utils.val_utils import AverageMeter, compute_psnr_ssim


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()  # MAE
        # self.loss_fn = nn.MSELoss()
        self.psnr = AverageMeter()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        temp_psnr, _, N = compute_psnr_ssim(restored, clean_patch)
        self.psnr.update(temp_psnr, N)
        self.log("val_loss", loss, sync_dist=True)
        return temp_psnr

    def on_validation_epoch_end(self):
        avg_psnr = self.psnr.avg
        self.psnr.reset()

        self.log('val_psnr', avg_psnr, sync_dist=True)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()[0]
        self.logger.experiment.add_scalar('lr', lr, self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150)
        # scheduler = WarmupCosineLR(optimizer=optimizer, max_iters=150)

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)

    log_dir = f"{'_'.join(opt.de_type)}/epoch{opt.epochs}_bs{opt.batch_size}_AdamW_lr{opt.lr}_patch{opt.patch_size}_ndup{opt.num_duplicate}_L1Loss"
    os.makedirs(f"logs/{log_dir}", exist_ok=True)
    os.makedirs(f"train_ckpt/{log_dir}", exist_ok=True)

    with open(f"logs/{log_dir}/train_args.txt", 'w') as f:
        f.write(str(opt))
    logger = TensorBoardLogger(save_dir=f"logs/{log_dir}")

    trainset = PromptTrainDataset(opt)
    trainset = Subset(trainset, range(100))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',                             # metric to monitor
        # directory to save checkpoints
        dirpath=f"train_ckpt/{log_dir}",
        filename='best-checkpoint_psnr{val_psnr:.2f}',  # checkpoint filename
        save_top_k=1,                                   # save only the best model
        mode='max'                                      # maximize val_psnr
    )

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    opt.derain_dir = 'data_split/Valid/Derain/'
    opt.desnow_dir = 'data_split/Valid/Desnow/'
    opt.num_duplicate = 1
    validset = PromptTrainDataset(opt)
    validloader = DataLoader(validset, batch_size=1,
                             pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    model = PromptIRModel()

    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus,
                         strategy="ddp_find_unused_parameters_true", logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader,
                val_dataloaders=validloader)


if __name__ == '__main__':
    main()
