from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

import cv2
import numpy as np
import os

class Free_solo_segmenter:
    def __init__(self): 
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"  
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, img_path):

        base_name = os.path.basename(img_path)
        renamed_name = f"{os.path.splitext(base_name)[0]}_mrcnn_segment{os.path.splitext(base_name)[1]}"

        image = cv2.imread(img_path)
        predictions = self.predictor(image)

        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        result_image = output.get_image()[:, :, ::-1]
        
        # Define the path where you want to save the image
        save_path = "free_solo_seg_mrcnn/uploads/" + renamed_name
        cv2.imwrite(save_path, result_image)
        print(f"Image saved to {save_path}")
        return save_path
