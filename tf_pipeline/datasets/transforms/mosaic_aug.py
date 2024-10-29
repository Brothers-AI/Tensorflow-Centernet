from typing import List, Dict, Any, Tuple, Sequence
import os
import json
import logging

import numpy as np
import cv2

from tf_pipeline.datasets.transforms.default import Transforms, TAG_NAME
from tf_pipeline.utils.registry.transforms import TRANSFORMS

LOG = logging.getLogger()

def coco_to_pascal(bbox):
    x, y, w, h = bbox
    bbox_pascal = [x, y, x + w, y + h]
    return bbox_pascal


@TRANSFORMS.register_module()
class BasicMosaicAugmentation(Transforms):
    def __init__(self, ann_file_path: str,
                 data_dir: str,
                 model_resolution: Sequence[int],
                 prob: float = 0.5):
        super(BasicMosaicAugmentation, self).__init__()
        self._name = "[BasicMosaicAugmentation]"
        assert os.path.exists(ann_file_path), f"Json file {ann_file_path} doesn't found. Please check"
        assert os.path.exists(data_dir), f"Data dir {data_dir} doesn't found. Please check"

        self.prob = prob

        LOG.info(f"{TAG_NAME} {self._name}: ann_file_path -> {ann_file_path}, data_dir -> {data_dir}, prob -> {prob}")

        # For constant random
        np.random.seed(42)

        self.data_initalized = False

        with open(ann_file_path, "r") as fp:
            json_data = json.load(fp)
        
        images = json_data['images']
        annotations = json_data['annotations']
        labels = json_data["categories"]

        self.model_resolution = model_resolution

        labels = {label["id"]: index for index, label in enumerate(labels)}

        records = {}
        for image in images:
            filepath = os.path.join(data_dir, image['file_name'])
            if not os.path.exists(filepath):
                raise AssertionError(f"{TAG_NAME} {self._name} Filepath: {filepath} not found in Mosaic Augmentation")
            records[image['id']] = {
                "image_id": image['id'],
                "bboxes": [],
                "labels": [],
                "filepath": filepath
            }
        
        for ann in annotations:
            bbox_xywh = ann['bbox']
            bbox_xyxy = coco_to_pascal(bbox_xywh)
            records[ann['image_id']]['bboxes'].append(bbox_xyxy)
            records[ann['image_id']]['labels'].append(labels[ann['category_id']])
        
        self.records = list(records.values())
        self.data_initalized = True
        self.samples_indices = list(range(0, len(self.records)))
    
    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            indices = [-1] + np.random.choice(self.samples_indices, size=3, replace=False).tolist()
            np.random.shuffle(indices)

            target_height, target_width = self.model_resolution

            mosaic_image = np.zeros((target_height * 2,
                                     target_width * 2, 3), dtype=np.uint8)
            
            yc, xc = (int(np.random.uniform(x//2, 3*x//2)) for x in (target_height, target_width))  # mosaic center x, y

            total_bboxes = []
            total_labels = []

            for idx, index in enumerate(indices):

                if index != -1:
                    local_record = self.records[index]
                    local_img = cv2.imread(local_record['filepath'])[:, :, :3]
                    local_img = cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB)

                    h, w = local_img.shape[:2]
                    local_bboxes = np.array(local_record['bboxes'], dtype=np.float32).reshape(-1, 4)
                    local_labels = np.array(local_record['labels'], dtype=np.int32).reshape(-1, 1)
                else:
                    h, w = image.shape[:2]
                    local_img = image

                    local_bboxes = bboxes.astype(np.float32).reshape(-1, 4)
                    local_labels = labels.astype(np.int32).reshape(-1, 1)
                
                if idx == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif idx == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif idx == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif idx == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(target_height * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                
                mosaic_image[y1a:y2a, x1a:x2a] = local_img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                padw = x1a - x1b
                padh = y1a - y1b

                local_bboxes[:, [0, 2]] += padw
                local_bboxes[:, [1, 3]] += padh

                total_bboxes.append(local_bboxes)
                total_labels.append(local_labels)
            
            bboxes = np.concatenate(total_bboxes, axis=0).reshape(-1, 4)
            labels = np.concatenate(total_labels, axis=0).reshape(-1, 1)

            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, 2 * target_width)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, 2 * target_height)

            area = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
            area = area[:, 0] * area[:, 1]
            indices_to_delete = area < 25

            bboxes = np.delete(bboxes, indices_to_delete, 0).astype(np.float32).reshape(-1, 4)
            labels = np.delete(labels, indices_to_delete, 0).astype(np.int32).reshape(-1, 1)
            return mosaic_image, bboxes, labels
        else:
            return image, bboxes, labels

def get_min_max_point(image_shape: Sequence[int], bboxes: np.ndarray):
    # Image Shape -> H, W
    # Bbox format -> xyxy
    H, W = image_shape
    min_x, min_y = 1, 1
    max_x, max_y = W - 1, H - 1
    if len(bboxes) > 0:
        min_x_bbox = np.min(bboxes[:, 0])
        min_y_bbox = np.min(bboxes[:, 1])
        max_x_bbox = np.max(bboxes[:, 2])
        max_y_bbox = np.max(bboxes[:, 3])

        min_x = int(max(min_x, min_x_bbox - 10))
        min_y = int(max(min_y, min_y_bbox - 10))
        max_x = int(min(max_x, max_x_bbox + 10))
        max_y = int(min(max_y, max_y_bbox + 10))
    return min_x, min_y, max_x, max_y


@TRANSFORMS.register_module()
class RoIMosaicAugmentation(Transforms):
    def __init__(self, ann_file_path: str,
                 data_dir: str,
                 model_resolution: Sequence[int],
                 prob: float = 0.5):
        super(RoIMosaicAugmentation, self).__init__()
        self._name = "[RoIMosaicAugmentation]"
        assert os.path.exists(ann_file_path), f"Json file {ann_file_path} doesn't found. Please check"
        assert os.path.exists(data_dir), f"Data dir {data_dir} doesn't found. Please check"

        self.prob = prob

        LOG.info(f"{TAG_NAME} {self._name}: ann_file_path -> {ann_file_path}, data_dir -> {data_dir}, prob -> {prob}")

        # For constant random
        np.random.seed(42)

        self.data_initalized = False

        with open(ann_file_path, "r") as fp:
            json_data = json.load(fp)
        
        images = json_data['images']
        annotations = json_data['annotations']
        labels = json_data["categories"]

        self.model_resolution = model_resolution

        labels = {label["id"]: index for index, label in enumerate(labels)}

        records = {}
        for image in images:
            filepath = os.path.join(data_dir, image['file_name'])
            if not os.path.exists(filepath):
                raise AssertionError(f"{TAG_NAME} {self._name} Filepath: {filepath} not found in Mosaic Augmentation")
            records[image['id']] = {
                "image_id": image['id'],
                "bboxes": [],
                "labels": [],
                "filepath": filepath
            }
        
        for ann in annotations:
            bbox_xywh = ann['bbox']
            bbox_xyxy = coco_to_pascal(bbox_xywh)
            records[ann['image_id']]['bboxes'].append(bbox_xyxy)
            records[ann['image_id']]['labels'].append(labels[ann['category_id']])
        
        self.records = list(records.values())
        self.data_initalized = True
        self.samples_indices = list(range(0, len(self.records)))
    
    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            indices = [-1] + np.random.choice(self.samples_indices, size=3, replace=False).tolist()
            np.random.shuffle(indices)

            target_height, target_width = self.model_resolution[0] // 2, self.model_resolution[1] // 2

            mosaic_image = np.zeros((target_height * 2,
                                     target_width * 2, 3), dtype=np.uint8)
            
            yc, xc = (int(np.random.uniform(x//2, 3*x//2)) for x in (target_height, target_width))  # mosaic center x, y

            total_bboxes = []
            total_labels = []

            for idx, index in enumerate(indices):

                if index != -1:
                    local_record = self.records[index]
                    local_img = cv2.imread(local_record['filepath'])[:, :, :3]
                    local_img = cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB)

                    h, w = local_img.shape[:2]
                    local_bboxes = np.array(local_record['bboxes'], dtype=np.float32).reshape(-1, 4)
                    local_labels = np.array(local_record['labels'], dtype=np.int32).reshape(-1, 1)
                else:
                    h, w = image.shape[:2]
                    local_img = image

                    local_bboxes = bboxes.astype(np.float32).reshape(-1, 4)
                    local_labels = labels.astype(np.int32).reshape(-1, 1)
                
                min_x, min_y, max_x, max_y = get_min_max_point([h, w], local_bboxes)
                local_img = local_img[min_y:max_y, min_x:max_x, :]
                h, w = local_img.shape[:2]
                local_bboxes[:, [0, 2]] -= min_x
                local_bboxes[:, [1, 3]] -= min_y
                
                if idx == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif idx == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif idx == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif idx == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(target_height * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                
                mosaic_image[y1a:y2a, x1a:x2a] = local_img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                padw = x1a - x1b
                padh = y1a - y1b

                local_bboxes[:, [0, 2]] += padw
                local_bboxes[:, [1, 3]] += padh

                total_bboxes.append(local_bboxes)
                total_labels.append(local_labels)
            
            bboxes = np.concatenate(total_bboxes, axis=0).reshape(-1, 4)
            labels = np.concatenate(total_labels, axis=0).reshape(-1, 1)

            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, 2 * target_width)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, 2 * target_height)

            area = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
            area = area[:, 0] * area[:, 1]
            indices_to_delete = area < 25

            bboxes = np.delete(bboxes, indices_to_delete, 0).astype(np.float32).reshape(-1, 4)
            labels = np.delete(labels, indices_to_delete, 0).astype(np.int32).reshape(-1, 1)
            return mosaic_image, bboxes, labels
        else:
            return image, bboxes, labels


@TRANSFORMS.register_module()
class RoIMosaicNMixUpAugmentation(Transforms):
    def __init__(self, ann_file_path: str,
                 data_dir: str,
                 model_resolution: Sequence[int],
                 prob: float = 0.5,
                 prob_mixup: float = 0.2):
        super(RoIMosaicNMixUpAugmentation, self).__init__()

        self._name = "[RoIMosaicNMixUpAugmentation]"
        assert os.path.exists(ann_file_path), f"Json file {ann_file_path} doesn't found. Please check"
        assert os.path.exists(data_dir), f"Data dir {data_dir} doesn't found. Please check"

        self.prob = prob
        self.prob_mixup = prob_mixup

        LOG.info(f"{TAG_NAME} {self._name}: ann_file_path -> {ann_file_path}, data_dir -> {data_dir}, prob -> {prob}, \
                 prob_mixup -> {prob_mixup}")

        # For constant random
        np.random.seed(42)

        self.data_initalized = False

        with open(ann_file_path, "r") as fp:
            json_data = json.load(fp)
        
        images = json_data['images']
        annotations = json_data['annotations']
        labels = json_data["categories"]

        self.model_resolution = model_resolution

        labels = {label["id"]: index for index, label in enumerate(labels)}

        records = {}
        for image in images:
            filepath = os.path.join(data_dir, image['file_name'])
            if not os.path.exists(filepath):
                raise AssertionError(f"{TAG_NAME} {self._name} Filepath: {filepath} not found in Mosaic Augmentation")
            records[image['id']] = {
                "image_id": image['id'],
                "bboxes": [],
                "labels": [],
                "filepath": filepath
            }
        
        for ann in annotations:
            bbox_xywh = ann['bbox']
            bbox_xyxy = coco_to_pascal(bbox_xywh)
            records[ann['image_id']]['bboxes'].append(bbox_xyxy)
            records[ann['image_id']]['labels'].append(labels[ann['category_id']])
        
        self.records = list(records.values())
        self.data_initalized = True
        self.samples_indices = list(range(0, len(self.records)))
    
    def get_mosaic(self, index, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray):
        indices = [index] + np.random.choice(self.samples_indices, size=3, replace=False).tolist()
        np.random.shuffle(indices)
        target_height, target_width = self.model_resolution[0] // 2, self.model_resolution[1] // 2
        mosaic_image = np.zeros((target_height * 2,
                                    target_width * 2, 3), dtype=np.uint8)
        
        yc, xc = (int(np.random.uniform(x//2, 3*x//2)) for x in (target_height, target_width))  # mosaic center x, y

        total_bboxes = []
        total_labels = []

        for idx, index in enumerate(indices):

            if index != -1:
                local_record = self.records[index]
                local_img = cv2.imread(local_record['filepath'])[:, :, :3]
                local_img = cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB)

                h, w = local_img.shape[:2]
                local_bboxes = np.array(local_record['bboxes'], dtype=np.float32).reshape(-1, 4)
                local_labels = np.array(local_record['labels'], dtype=np.int32).reshape(-1, 1)
            else:
                h, w = image.shape[:2]
                local_img = image

                local_bboxes = bboxes.astype(np.float32).reshape(-1, 4)
                local_labels = labels.astype(np.int32).reshape(-1, 1)
            
            min_x, min_y, max_x, max_y = get_min_max_point([h, w], local_bboxes)
            local_img = local_img[min_y:max_y, min_x:max_x, :]
            h, w = local_img.shape[:2]
            local_bboxes[:, [0, 2]] -= min_x
            local_bboxes[:, [1, 3]] -= min_y
            
            if idx == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(target_height * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            mosaic_image[y1a:y2a, x1a:x2a] = local_img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            local_bboxes[:, [0, 2]] += padw
            local_bboxes[:, [1, 3]] += padh

            total_bboxes.append(local_bboxes)
            total_labels.append(local_labels)
        
        bboxes = np.concatenate(total_bboxes, axis=0).reshape(-1, 4)
        labels = np.concatenate(total_labels, axis=0).reshape(-1, 1)

        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, 2 * target_width)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, 2 * target_height)

        area = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]
        area = area[:, 0] * area[:, 1]
        indices_to_delete = area < 25

        bboxes = np.delete(bboxes, indices_to_delete, 0).astype(np.float32).reshape(-1, 4)
        labels = np.delete(labels, indices_to_delete, 0).astype(np.int32).reshape(-1, 1)
        return mosaic_image, bboxes, labels
    
    def mixup(self, image_1, bboxes_1, labels_1, image_2, bboxes_2, labels_2):
        '''Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf.'''
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        mixup_image = (image_1 * r + image_2 * (1 - r)).astype(np.uint8)
        mixup_bboxes = np.concatenate((bboxes_1, bboxes_2), axis=0)
        mixup_labels = np.concatenate((labels_1, labels_2), axis=0)
        return mixup_image, mixup_bboxes, mixup_labels

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() < self.prob:
            
            # First Mosaic
            mosaic_image_1, bboxes_1, labels_1 = self.get_mosaic(-1, image, bboxes, labels)

            # Mixup
            if np.random.random() < self.prob_mixup:
                index = np.random.randint(0, len(self.records))
                mosaic_image_2, bboxes_2, labels_2 = self.get_mosaic(index, image, bboxes, labels)

                mixup_image, mixup_bboxes, mixup_labels = self.mixup(mosaic_image_1, bboxes_1, labels_1,
                                                                     mosaic_image_2, bboxes_2, labels_2)
                mixup_bboxes = np.array(mixup_bboxes, dtype=np.float32).reshape(-1, 4)
                mixup_labels = np.array(mixup_labels, dtype=np.int32).reshape(-1, 1)
                return mixup_image, mixup_bboxes, mixup_labels
            else:
                bboxes_1 = np.array(bboxes_1, dtype=np.float32).reshape(-1, 4)
                labels_1 = np.array(labels_1, dtype=np.int32).reshape(-1, 1)
                return mosaic_image_1, bboxes_1, labels_1
        else:
            return image, bboxes, labels