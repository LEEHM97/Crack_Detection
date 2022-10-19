import cv2
import torch
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