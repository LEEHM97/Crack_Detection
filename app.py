import cv2
import numpy as np
import os
import torch

from flask import Flask
from flask import render_template
from flask import request
from flask import session
from werkzeug.utils import secure_filename

from models import SegmentationModel
from transforms import make_transform
import utils

model_dir = "./checkpoints/unet++_f1=0.8276.ckpt"
args = utils.basic_args()
_, test_transform = make_transform(args)

model = SegmentationModel.load_from_checkpoint(model_dir, args=args)
model.freeze()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'outputs')
app.secret_key = 'cracker'

@app.route('/')
def hello():
    return render_template('image_upload.html')

@app.route('/LandMark', methods = ['GET', 'POST'])
def Cls_LandMark():
    if request.method == 'POST':
        img = request.files['chooseFile']
        img_name = secure_filename(img.filename)
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        # img_str = img.read()
        
        # img_bytes = np.fromstring(img_str, dtype = np.uint8)
        # decode_img = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
        image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        
        transformed_image = test_transform(image = image)
        transformed_image = transformed_image['image'].float()
        
        preprocessed_image = transformed_image.unsqueeze(0)
        outputs = model(preprocessed_image)
        output = torch.sigmoid(outputs).squeeze(0).squeeze(0).numpy()
        output = np.where(output<0.5, 0, 255)
        
        cv2.imwrite('./static/outputs/predicted_mask.png', output)
        

        skel = utils.skeletonize(output)
        canny = utils.canny(output)
        
        pixel_pairs, distance = utils.get_width(skel, canny)
        max_width_idx, max_width = utils.get_max_width(distance)
        
        width_img = utils.visualize_max_width(pixel_pairs, max_width_idx, max_width, canny)

        contour_skel = utils.contour_skel(output)
        contours = utils.get_contour(output, contour_skel)
        
        area_img = utils.visuzlize_contour_area(output, contours)
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_mask.png')
        width_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'viz_max_width.png')
        area_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'viz_contour_area.png')
        
        return render_template('result_page.html', original=image_path, output=output_path, width_img=width_img_path, area_img=area_img_path)
    
    else:
        return render_template('image_upload.html')

app.run(host='0.0.0.0', port=5000)