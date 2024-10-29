from typing import List, Any, Dict
import json
import logging
import os

import numpy as np

from tf_pipeline.utils.metrics import process_batch

TAG_NAME = "[CustomEvaluation]"
LOG = logging.getLogger()


class CustomEvaluation(object):
    def __init__(self, val_json_path: str, save_dir: str, plot_curve: bool = True):
        
        self.stats = []
        self.ap = []

        self.iouv = np.linspace(0.5, 0.95, 10)
        self.niou = len(self.iouv)

        self.is_update_state_called = False

        # read the json
        with open(val_json_path, "r") as fp:
            json_data = json.load(fp)
        
        cats = json_data['categories']
        self.cat_names = [cat['name'] for cat in cats]

        self.save_dir = save_dir
        # create directory
        os.makedirs(save_dir, exist_ok=True)

        self.plot_curve = plot_curve

        self.seen = 0
    
    def reset_data(self):
        self.is_update_state_called = False
        self.stats.clear()
        self.seen = 0
        return

    def update_state(self, detections: List[Any], gt_bboxes: List[Any], gt_labels: List[Any]):
        # detections -> List[x1, y1, x2, y2, score, label]

        detections_np = np.array(detections).reshape(-1, 6)
        gt_bboxes_np = np.array(gt_bboxes).reshape(-1, 4)
        gt_labels_np = np.array(gt_labels).reshape(-1, 1)

        labels = np.hstack((gt_labels_np, gt_bboxes_np))
        correct = process_batch(detections_np, labels, self.iouv)

        # Valid bboxes
        valid_bbox_indices = np.where((gt_bboxes_np[:, 2] - gt_bboxes_np[:, 0]) * (gt_bboxes_np[:, 3] - gt_bboxes_np[:, 1]) != 0)[0]

        # Append statistics (correct, conf, pcls, tcls)
        self.stats.append((correct, detections_np[:, -2], detections_np[:, -1], gt_labels_np[valid_bbox_indices][:, 0].tolist()))

        self.seen += 1

        self.is_update_state_called = True
        return
    
    def result(self):

        if not self.is_update_state_called:
            LOG.error(f"{TAG_NAME} update_state is not called.")
            return

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy

        if len(stats) and stats[0].any():

            from tf_pipeline.utils.metrics import ap_per_class
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plot_curve, save_dir=self.save_dir, names=self.cat_names)

            AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() -1

            LOG.info(f"{TAG_NAME} IoU@0.5 best mF1 threshold near {AP50_F1_max_idx/1000.0}")
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=len(self.cat_names))  # number of targets per class

            # Print results
            s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
            LOG.info(s)
            pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
            LOG.info(pf % ('all', self.seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

            self.pr_metric_result = (map50, map)
            map_wighted = 0.0
            for i, c in enumerate(ap_class):
                map_wighted += (nt[c] * ap50[i])
                LOG.info(pf % (self.cat_names[c], self.seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                    f1[i, AP50_F1_max_idx], ap50[i], ap[i]))
            
            LOG.info(f"{TAG_NAME} map_weighted @ 0.5: {map_wighted / nt.sum()}")
            
        return