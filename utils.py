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


def calc_distance(a, b):
    y_distance = (a[0] - b[0]) ** 2
    x_distance = (a[1] - b[1]) ** 2
    return  x_distance + y_distance


def get_width(skel, canny):
    pixel_pairs = []
    distance = []
    
    white_pixels_skel= list(map(tuple, np.argwhere(skel==255)))
    white_pixels_canny= list(map(tuple, np.argwhere(canny==255)))
    
    for skel_idx in range(len(white_pixels_skel)):
        
        min_distance = math.inf
        temp_pixel_pairs = None

        for canny_idx in range(len(white_pixels_canny)):
            temp_distance = calc_distance(white_pixels_skel[skel_idx], white_pixels_canny[canny_idx])

            if  temp_distance < min_distance:
                temp_pixel_pairs = [white_pixels_skel[skel_idx], white_pixels_canny[canny_idx]]
                min_distance = temp_distance

        pixel_pairs.append(temp_pixel_pairs)
        distance.append(math.sqrt(min_distance))
    
    return pixel_pairs, distance
    
        
def get_max_width(distance):
    max_width = -math.inf
    cnt = 0
    max_width_idx = 0
    
    for w in distance:
        cnt += 1
        
        if w > max_width:
            max_width = w
            max_width_idx = cnt-1
    
    return max_width_idx, int(max_width*2)


def visualize_width(output, pixel_pairs, max_width_idx):
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    viz_image = np.zeros(output.shape, dtype=np.uint8)
    
    for skel_idx, canny_idx in pixel_pairs:
        viz_image[skel_idx] = np.array([0, 0, 255])
        viz_image[canny_idx] = np.array([0, 255, 0])
        
    viz_image[pixel_pairs[max_width_idx][0]] = np.array([255, 0, 0])
    viz_image[pixel_pairs[max_width_idx][0]] = np.array([255, 0, 0])
    
    cv2.imwrite('./outputs/viz_width.png', viz_image)
    
def visualize_max_width(pixel_pairs, max_width_idx, max_width, canny):
    y1 = pixel_pairs[max_width_idx][0][0]
    x1 = pixel_pairs[max_width_idx][0][1]
    y2 = pixel_pairs[max_width_idx][1][0]
    x2 = pixel_pairs[max_width_idx][1][1]
    
    x3 = (2 * x1) - x2
    y3 = (2 * y1) - y2
    
    viz_box = canny.copy()
    viz_box = cv2.cvtColor(viz_box, cv2.COLOR_GRAY2BGR)
    
    txt = f"width: {max_width}"
    
    viz_box = cv2.line(viz_box, (x2,y2), (x3, y3), (255,0,0), 3)
    cv2.putText(img=viz_box, text=txt, org=(x2, y2), fontFace=1, fontScale=2, thickness=3, color=(255, 0, 0))
    
    cv2.imwrite('./outputs/viz_max_width.png', viz_box)