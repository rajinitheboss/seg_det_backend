from ultralytics import YOLO
import os
from PIL import Image
import shutil
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

class yolov8:
    # def add_title_to_image(self,image_path, title_text):
    #     img = mpimg.imread(image_path)
    #     plt.imshow(img)
    #     plt.title(title_text, color='black', fontsize=15)
    #     plt.axis('off')
    #     output_path = image_path
    #     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    def detect(image_path):
        # yolo=yolov8()
        model = YOLO('yolov8s-seg.pt')
        base_name = os.path.basename(image_path)

        print(image_path)

        print(base_name)
        # object detection using yolov8

        # image_path = os.path.join('uploads',base_name)
        results = model.predict(image_path,save=True,save_txt=True)

        # Renaming the predicted file
        renamed_name = f"{os.path.splitext(base_name)[0]}_yolov8_detect{os.path.splitext(base_name)[1]}"

        # renaming and moving to the uploads folder 
        os.rename('runs/segment/predict/'+base_name,'runs/segment/predict/'+renamed_name)
        shutil.move('runs/segment/predict/'+renamed_name,'yolov8/uploads/')

        # deleting the generated directory

        shutil.rmtree('runs')
        # yolo.add_title_to_image('yolov8/'+ 'uploads/' + renamed_name,"Yolov8")

        return 'yolov8/'+ 'uploads/' + renamed_name