from flask import Flask, request, jsonify,send_file,make_response
from werkzeug.utils import secure_filename
from PIL import Image
import os
from yolov8.yolov8 import yolov8
from flask_cors import CORS
import shutil
from datetime import datetime
from sam.sam import SegmentAnythingAPI,SegmentAnythingCLI

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return 'Hello, World!'


def segment_image_sam(image_path):
    SegmentAnythingCLI.segment_image(image_path=image_path)


@app.route('/upload_segment',methods=['POST'])
def upload_file_segment():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        original_ext = file.filename.rsplit('.', 1)[1].lower()  
        new_filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.{original_ext}"  
        file_path = os.path.join('sam',app.config['UPLOAD_FOLDER'], new_filename)
        
       
        file.save(file_path)
        sam_segmenter = SegmentAnythingCLI()
        segmented_image_path = sam_segmenter.segment_image(image_path=file_path)

        print('segmentation and moving to uploads folder is done')


        return send_file(segmented_image_path, mimetype='image/png'),200

    else:
        return jsonify({'error': 'Invalid file format'}), 400



@app.route('/upload_detect', methods=['POST'])
def upload_file_detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        if os.path.isdir('runs'):
            shutil.rmtree('runs')
        original_ext = file.filename.rsplit('.', 1)[1].lower()  
        new_filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.{original_ext}"  
        file_path = os.path.join('yolov8',app.config['UPLOAD_FOLDER'], new_filename)
        
       
        file.save(file_path)
        detected_image_path = detect_image_yolov8(image_path=file_path)

        print('detection and moving to uploads folder is done')


        return send_file(detected_image_path, mimetype='image/png'),200

    else:
        return jsonify({'error': 'Invalid file format'}), 400


def detect_image_yolov8(image_path):
    detected_image_path = yolov8.detect(image_path=image_path)
    return detected_image_path


def process_image(image_path):
    # Example processing: Resize the image to 100x100
    with Image.open(image_path) as img:
        img.thumbnail((100, 100))
        processed_path = f"{os.path.splitext(image_path)[0]}_thumbnail{os.path.splitext(image_path)[1]}"
        img.save(processed_path)
    return processed_path


if __name__ == '__main__':
    app.run(debug=True)