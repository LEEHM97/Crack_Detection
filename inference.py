import os
import argparse
import torch
import cv2

import numpy as np
import pytorch_lightning as pl

import utils
from transforms import make_transform
from models import SegmentationModel

parser = argparse.ArgumentParser()

# model 설정

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--project", type=str, default="Crack_Detection")
parser.add_argument("--name", type=str, default="unet++_b1")
parser.add_argument("--model", type=str, default="DeepLabV3Plus")
parser.add_argument("--encoder", type=str, default="efficientnet-b4")
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
    image_dir = "./Datasets/Test/images/20160222_115305.jpg"
    model_dir = "./checkpoints/deeplabv3+_b4.ckpt"
    
    _, test_transform = make_transform(args)
    model = SegmentationModel.load_from_checkpoint(model_dir, args=args)
    model.freeze()

    output = utils.get_output(image_dir, model, test_transform)
    
    cv2.imwrite('./static/outputs/predicted_mask.png', output)
    
    skel = utils.skeletonize(output)
    canny = utils.canny(output)
    
    cv2.imwrite('./static/outputs/skel.png', skel)
    cv2.imwrite('./static/outputs/canny.png', canny)
    
    pixel_pairs, distance = utils.get_width(skel, canny)
    max_width_idx, max_width = utils.get_max_width(distance)
    
    utils.visualize_width(output, pixel_pairs, max_width_idx)
    utils.visualize_max_width(pixel_pairs, max_width_idx, max_width, canny)
    
    contour_skel = utils.contour_skel(output)
    contours = utils.get_contour(output, contour_skel)
    
    utils.visuzlize_contour_area(output, contours)