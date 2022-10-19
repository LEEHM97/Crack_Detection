import cv2
import torch
import math
import numpy as np

from transforms import make_transform

def get_output(image_dir, model, transform):
    image = cv2.imread(image_dir)
    
    transformed_image = transform(image=image)
    transformed_image = transformed_image['image'].float()
    
    preprocessed_image = transformed_image.unsqueeze(0)
    outputs = model(preprocessed_image)
    output = torch.sigmoid(outputs).squeeze(0).squeeze(0).numpy()
    output = np.where(output<0.5, 0, 255)
    
    return output

def skeletonize(image):
    image = image.astype(np.uint8)
    skel = cv2.ximgproc.thinning(image)
    return skel

def canny(image):
    image = image.astype(np.uint8)
    canny = cv2.Canny(image, 100, 150)
    return canny