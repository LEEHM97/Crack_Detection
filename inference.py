import os
import argparse
import torch
import cv2

import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

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
parser.add_argument("--name", type=str, default="unet++_b1")
parser.add_argument("--model", type=str, default="UnetPlusPlus")
parser.add_argument("--encoder", type=str, default="efficientnet-b1")
parser.add_argument("--precision", type=int, default=16)


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

if __name__ == "__main__":
    image_dir = "./Datasets/Test/101_3e7a208a-e38a-41ad-a46c-a3a27addd5d9.jpg"
    model_dir = "./checkpoints/unet++_f1=0.8276.ckpt"
    
    _, test_transform = make_transform(args)
    model = SegmentationModel.load_from_checkpoint(model_dir, args=args)
    model.freeze()
    
    trainer = pl.Trainer(accelerator='gpu',
                        devices=1,
                        max_epochs=1,
                        log_every_n_steps=1)    
    
    image = cv2.imread(image_dir)
    
    transformed_image = test_transform(image=image)
    transformed_image = transformed_image['image'].float()
    
    preprocessed_image = transformed_image.unsqueeze(0)
    outputs = model(preprocessed_image)
    output = torch.sigmoid(outputs).squeeze(0).squeeze(0).numpy()
    output = np.where(output<0.6, 0, 255)
    
    cv2.imwrite('./outputs/predicted_mask.png', output)
    
    output = np.array(output, dtype=np.uint8)
    output = cv2.resize(output, (1000, 600))
    
    cv2.imshow('predicted', output)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()