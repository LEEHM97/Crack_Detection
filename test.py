import os
import argparse
import torch
import wandb
import glob
import natsort
import cv2

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
parser.add_argument("--idx", type=int, default=0)


# training 설정

parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--kfold", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--scheduler", type=str, default="reducelr")
parser.add_argument("--crop_image_size", type=int, default=320)

parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default = './config/cfg.yml',
                    type=str)
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()
train_data_dir = "./Datasets/Train"
test_data_dir = "./Datasets/Test"

if __name__ == "__main__":
    pl.seed_everything(args.seed)
    train_images = np.array(natsort.natsorted(glob.glob(os.path.join(train_data_dir, "images", "*"))))
    train_masks = np.array(natsort.natsorted(glob.glob(os.path.join(train_data_dir, "masks", "*"))))

    test_images = np.array(natsort.natsorted(glob.glob(os.path.join(test_data_dir, "images", "*"))))
    test_masks = np.array(natsort.natsorted(glob.glob(os.path.join(test_data_dir, "masks", "*"))))
    
    train_transform, test_transform = make_transform(args)
    model = SegmentationModel(args)
    # model = SegmentationModel.load_from_checkpoint('checkpoints\\unet++_b0_fold02_val\\f1=0.0000-v2.ckpt', args)
    
    model.freeze()
    test_ds = SegmentationDataset(test_images, test_masks, test_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, num_workers=args.num_workers)
    
    trainer = pl.Trainer(accelerator='gpu',
                        devices=1,
                        max_epochs=5,
                        log_every_n_steps=1)
    idx = args.idx

    img, mask = test_ds[idx]
    new_img = img.unsqueeze(0)

    outputs = model(new_img)
    
    origin_output = torch.sigmoid(outputs).squeeze(0).squeeze(0).numpy()
    output = np.where(origin_output<0.6, 0, 255) 
    outputs = outputs.numpy()
    
    cv2.imwrite('./outputs/origin_output.png', origin_output)
    cv2.imwrite('./outputs/bw_output.png', output)