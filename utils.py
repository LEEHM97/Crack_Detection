import cv2
import torch
import math
import argparse
import numpy as np

from transforms import make_transform

CLASS_COLOR = np.array([[0, 0, 0], [192, 0, 128],[0, 128, 192],[0, 128, 64],[128, 0, 0],
        [64, 0, 128],[64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],
        [128, 0, 192],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 128, 128],
        [153, 0, 51], [102, 255, 153] , [255, 51, 153], [102, 204, 255],[0, 102, 51],
        [255, 153, 204]], np.uint8)

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
    
    cv2.imwrite('./static/outputs/viz_width.png', viz_image)
    return viz_image
    
    
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
    
    viz_box = cv2.line(viz_box, (x2,y2), (x3, y3), (255,255,255), 4)
    cv2.putText(img=viz_box, text=txt, org=(x2, y2), fontFace=1, fontScale=4, thickness=3, color=(255, 255, 255))
    
    cv2.imwrite('./static/outputs/viz_max_width.png', viz_box)
    return viz_box

def contour_skel(output):
    output = output.astype(np.uint8)
    
    # Threshold the image
    kernel = np.ones((3, 3), np.uint8)

    output = dilation = cv2.dilate(output, kernel, iterations=20)
    _ ,output = cv2.threshold(output, 127, 255, 0)

    # Step 1: Create an empty skeleton
    skel = np.zeros(output.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(output, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(output, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(output, element)
        skel = cv2.bitwise_or(skel,temp)
        output = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(output)==0:
            break
        
    return skel

def get_contour(output, skel):
    areas = []
    output = output.astype(np.uint8)
    
    _, thresh = cv2.threshold(output, 160, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_or(skel, thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    contours = list(filter(lambda x: cv2.contourArea(x) > 200, contours))

    return contours

def visuzlize_contour_area(output, contours):
    output = output.astype(np.uint8)
    
    viz_box = output.copy()
    viz_box = cv2.cvtColor(viz_box, cv2.COLOR_GRAY2BGR)
    
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        viz_box = cv2.rectangle(viz_box, (x,y), (x+w, y+h), CLASS_COLOR[idx+1].tolist(), 5)
        viz_box = cv2.drawContours(viz_box, contours, idx ,CLASS_COLOR[idx+1].tolist(), 5)

        txt = f"{idx}area: {cv2.contourArea(contours[idx])}"
        cv2.putText(img=viz_box, text=txt, org=(x+10, y+40), fontFace=1, fontScale=4, thickness=4, color=(255, 255, 255))
    
    cv2.imwrite('./static/outputs/viz_contour_area.png', viz_box)
    return viz_box


def basic_args():
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
    
    return args