from ultralytics import YOLO
import os
from PIL import Image
import shutil

class yolov8:
    def detect(image_path):
        model = YOLO('yolov8s-seg.pt')
        base_name = os.path.basename(image_path)

        # object detection using yolov8
        results = model.predict(image_path,save=True,save_txt=True)

        # Renaming the predicted file
        renamed_name = f"{os.path.splitext(base_name)[0]}_yolov8_detect{os.path.splitext(base_name)[1]}"

        # renaming and moving to the uploads folder 
        os.rename('runs/segment/predict/'+base_name,'runs/segment/predict/'+renamed_name)
        shutil.move('runs/segment/predict/'+renamed_name,'uploads/')

        # deleting the generated directory

        shutil.rmtree('runs')

        return 'uploads/' + renamed_name