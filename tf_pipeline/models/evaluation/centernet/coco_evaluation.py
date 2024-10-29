import os
import logging
from typing import List, Dict, Any
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

LOG = logging.getLogger()
TAG_NAME = "[Evaluation]"

class COCOEvaluation(object):
    def __init__(self, val_json_path: str):

        assert os.path.exists(val_json_path), f"Json path {val_json_path} not found. Please check"

        self._name = "[COCO]"
        LOG.info(f"{TAG_NAME} {self._name} val_json_path: {val_json_path}")

        self.coco_org = COCO(val_json_path)

        categories = self.coco_org.cats
        self.pred_to_cat = {idx: cat_id for idx, cat_id in enumerate(categories.keys())}

        self.is_update_state_called = False

        self.detections: List[Dict[str, Any]] = []

        # Create a local directory for storing the data
        self.tmp_dir = ".temp"
        os.makedirs(self.tmp_dir, exist_ok=True)
    
    def reset_data(self):
        self.detections.clear()
        self.is_update_state_called = False
        return
    
    def update_state(self, detections: List[Any], image_id: int):
        # detections -> List[x1, y1, x2, y2, score, label]
        for det in detections:
            if isinstance(det, np.ndarray):
                x1, y1, x2, y2, score, label = det.tolist()
            else:
                x1, y1, x2, y2, score, label = det
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            data = {
                "image_id": int(image_id),
                "category_id": int(self.pred_to_cat[int(label)]),
                "bbox": bbox_xywh,
                "score": score
            }

            # Store the data
            self.detections.append(data)

        self.is_update_state_called = True
        return
    
    def result(self):
        if not self.is_update_state_called:
            raise RuntimeError("Please call \"update_state\" API to store the detections for \
                               each image in validation / test set.")
        results_json = os.path.join(self.tmp_dir, "results.json")
        with open(results_json, "w") as fp:
            json.dump(self.detections, fp, indent=4)
        
        coco_detections = self.coco_org.loadRes(results_json)
        coco_eval = COCOeval(self.coco_org, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        os.system(f"rm -r {self.tmp_dir}")
        return