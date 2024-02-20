import os
import cv2
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


HOME = os.getcwd()
print("HOME:", HOME)

CHECKPOINT_PATH = os.path.join(HOME,'sam', "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))


class SegmentAnythingAPI:
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path or self._default_checkpoint_path()
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path).to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def _default_checkpoint_path(self):
        home = os.getcwd()
        return os.path.join(home, "sam/weights", "sam_vit_h_4b8939.pth")

    def segment_image(self, image_path):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_result = self.mask_generator.generate(image_rgb)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        return annotated_image

class SegmentAnythingCLI:
    def __init__(self):
        self.api = SegmentAnythingAPI()

    def segment_image(self, image_path):
        segmented_image = self.api.segment_image(image_path)

        base_name = os.path.basename(image_path)
        renamed_name = f"{os.path.splitext(base_name)[0]}_sam_segment{os.path.splitext(base_name)[1]}"
        cv2.imwrite("sam/uploads/"+renamed_name, segmented_image)
        # print('something happend')
        return 'sam/' + 'uploads/' + renamed_name

# Example usage:
if __name__ == "__main__":
    cli = SegmentAnythingCLI()
    cli.segment_image("data/dog.jpeg")