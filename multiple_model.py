from flask import Blueprint, jsonify
from flask import Flask, request, jsonify,send_file,make_response
from werkzeug.utils import secure_filename
from PIL import Image
import os
from yolov8.yolov8 import yolov8
from flask_cors import CORS
import shutil
from datetime import datetime
from sam.sam import SegmentAnythingAPI,SegmentAnythingCLI
from free_solo_seg_mrcnn.segment import Free_solo_segmenter
from free_solo_det_fast_rcnn.detect import Free_solo_detector
from detr.detr import DETRDetector
import threading
from CA_Net.show_fused_heatmap import ImageProcessor
import zipfile
import io
import base64
from SAHI.sahi_prediction import sahiDetector


multiple_model = Blueprint('multiple_model',__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

threading_result = dict()



def sam_segmentation(file_path):
    sam_segmenter = SegmentAnythingCLI()
    segmented_image_path = sam_segmenter.segment_image(image_path=file_path)
    return segmented_image_path

def mrcnn_segmentation(file_path):
    freeSoloSegmenter = Free_solo_segmenter()
    print('MRCNN SEGMENTATION')
    segmented_image_path = freeSoloSegmenter.onImage(img_path=file_path)
    return segmented_image_path

def canet_segmentation(file_path):
    print("CANET Segmentation")
    segmented_image_path = ImageProcessor.segmentImage(filepath=file_path) 
    return segmented_image_path


def detect_image_yolov8(image_path):
    detected_image_path = yolov8.detect(image_path=image_path)
    return detected_image_path

def yolov8_detection(file_path):
    print("yolov8 detection")
    detected_image_path = detect_image_yolov8(image_path=file_path)
    return detected_image_path

def free_solo_detection(file_path):
    freeSoloDetector = Free_solo_detector()
    detected_image_path = freeSoloDetector.onImage(img_path=file_path)
    return detected_image_path

def detr_detection(file_path):
    detrDetector = DETRDetector()
    detected_image_path = detrDetector.predictImage(image_path = file_path)
    return detected_image_path

def sahi_detection(file_path):
    detected_image_path = sahiDetector.DetectImage(filepath=file_path)
    return detected_image_path




@multiple_model.route('/multiple_model',methods=['POST'])
def run_multiple_models():

    threading_result.clear()

    if 'file' not in request.files:
        return jsonify({'error':'No file part'}),400
    
    file = request.files['file']
    
    original_ext = file.filename.rsplit('.', 1)[1].lower()  
    new_filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.{original_ext}"  
    file_path = os.path.join('uploads', new_filename)
    file.save(file_path)

    result = dict()

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):

        if 'model1' in request.form :
            model1 = request.form['model1'].strip().lower()
            print(model1)
        else:
            model1 = ''
        
        if 'model2' in request.form:
            model2 = request.form['model2'].strip().lower()
            print(model2)
        else:
            model2 = ''
        
        if 'model3' in request.form:
            model3 = request.form['model3'].strip().lower()
        else:
            model3 = ''

        model_dict = {'sam':0,'m_rcnn':1,'yolov8':2,'free solo':3,'detr':4,'sahi':5}
        arr = [sam_segmentation,mrcnn_segmentation,yolov8_detection,free_solo_detection,detr_detection,sahi_detection]

        if model1 in model_dict:
            temp = arr[model_dict[model1]](file_path)
            result['model1'] = temp
            print(temp)
        
        if model2 in model_dict:
            temp = arr[model_dict[model2]](file_path)
            result['model2'] = temp
            print(temp)
        

        if model3 in model_dict:
            temp = arr[model_dict[model3]](file_path)
            result['model3'] = temp
            print(temp)
        

    local_dict = dict(threading_result)

    threading_result.clear()

    data_to_be_sent = dict()
    # for model in local_dict:
    #     t = 0
    #     if model == request.form['model1']:
    #         t = 1
    #     elif model == request.form['model2']:
    #         t = 2
    #     elif model == request.form['model3']:
    #         t = 3
    #     with open(local_dict[model],'rb') as f:
    #         data = f.read()
    #         image_base64 = base64.b64encode(data).decode('utf-8')
    #         data_to_be_sent['model'+str(t)]= image_base64

    print(result)

    for model in result:
        with open(result[model],'rb') as f:
            data = f.read()
            image_base64 = base64.b64encode(data).decode('utf-8')
            data_to_be_sent[model] = image_base64 

    return jsonify(data_to_be_sent)




@multiple_model.route('/get')
def send():
    return 'krishna'