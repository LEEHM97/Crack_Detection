import os
import argparse
import torch
import wandb
import glob

import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from datasets import SegmentationDataset
from transforms import make_transform
from models import SegmentationModel

parser = argparse.ArgumentParser()

# model 설정

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--project", type=str, default="Crack_Detection")
parser.add_argument("--name", type=str, default="unet++_b0")
parser.add_argument("--model", type=str, default="UnetPlusPlus")
parser.add_argument("--encoder", type=str, default="efficientnet-b0")
parser.add_argument("--precision", type=int, default=16)


# training 설정

parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--kfold", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--scheduler", type=str, default="recudelr")
parser.add_argument("--crop_image_size", type=int, default=320)

args = parser.parse_args()
train_data_dir = "./Datasets/Train"
test_data_dir = "./Datasets/Test"

if __name__ == "__main__":
    pl.seed_everything(args.seed)
    train_images = np.array(sorted(glob.glob(os.path.join(train_data_dir, "images", "*"))))
    train_masks = np.array(sorted(glob.glob(os.path.join(train_data_dir, "masks", "*"))))

    test_images = np.array(sorted(glob.glob(os.path.join(test_data_dir, "images", "*"))))
    test_masks = np.array(sorted(glob.glob(os.path.join(test_data_dir, "masks", "*"))))

    kf = KFold(n_splits=args.kfold)
    for idx, (train_index, val_index) in enumerate(kf.split(X=train_images)):
        wandb_logger = WandbLogger(
            project=args.project,
            name=f"{args.name}_fold{idx + 1:02d}",
            entity="crack_detection_22",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val/jaccard_index_value",
            dirpath="checkpoints",
            filename=f"{args.name}_fold{idx + 1:02d}_"
            + "{val/jaccard_index_value:.4f}",
            save_top_k=3,
            mode="max",
        )
    early_stop_callback = EarlyStopping(
        monitor="val/loss", min_delta=0.00, patience=50, verbose=True, mode="min"
    )

    train_transform, test_transform = make_transform(args)
    model = SegmentationModel(args)

    train_ds = SegmentationDataset(
        train_images[train_index], train_masks[train_index], train_transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    val_ds = SegmentationDataset(
        train_images[val_index], train_masks[val_index], train_transform
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    test_ds = SegmentationDataset(test_images, test_masks, test_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, num_workers=args.num_workers
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(dataloaders=test_dataloader)

    wandb.finish()
