from typing import Tuple, Dict, Sequence
import math
import os
import json
import logging

import cv2
import numpy as np
import tensorflow as tf

from tf_pipeline.datasets.transforms import Compose, ResizeTo, BasicMosaicAugmentation, RoIMosaicAugmentation, \
                                            ResizeAugV1, ResizeAugV2
from tf_pipeline.datasets.default import DefaultDatasetV2
from tf_pipeline.datasets.utils import gaussian_radius, draw_umich_gaussian
from tf_pipeline.utils.registry.datasets import DATASETS
from tf_pipeline.utils.registry.transforms import TRANSFORMS

LOG = logging.getLogger()
TAG_NAME = '[Dataset]'


def coco_to_pascal(bbox):
    x, y, w, h = bbox
    bbox_pascal = [x, y, x + w, y + h]
    return bbox_pascal


@DATASETS.register_module()
class CocoDatasetV2(DefaultDatasetV2):
    def __init__(self, root_dir: str, json_file_name: str, split_suffix: str,
                 model_resolution: Sequence[np.int], split: np.str, transforms: Dict,
                 init_lr: float = None, max_objects: int = 100, down_ratio: int = 4,
                 batch_size: np.int = 1, shuffle: np.bool = False):

        self._name = '[CocoDatasetV2]'

        self.model_resolution = model_resolution

        img_dir = os.path.join(root_dir, split_suffix)
        json_path = os.path.join(root_dir, "annotations", json_file_name)

        LOG.info(f"{TAG_NAME} {self._name} split: {split}")

        self.records, self.labels, self._class_names = self.load_json(
            json_path, img_dir)
        
        self._json_path = json_path

        self.out_w = self.model_resolution[1]
        self.out_h = self.model_resolution[0]

        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([127, 127, 127], dtype=np.float32)

        self.max_objects = int(max_objects)

        self.down_ratio = float(down_ratio)
        self.heat_h, self.heat_w = int(self.out_h // self.down_ratio), int(self.out_w // self.down_ratio)

        self.initalize_augmentations(transforms, img_dir)

        self.dataset_len = len(self.records)

        super(CocoDatasetV2, self).__init__(batch_size, shuffle)

    @property
    def num_classes(self):
        return len(self.labels)
    
    @property
    def val_json_path(self):
        return self._json_path

    @property
    def class_names(self):
        return self._class_names

    def load_json(self, filepath, img_dir):
        with open(filepath, "r") as fp:
            json_data = json.load(fp)

        labels = json_data["categories"]
        class_names = {index: label['name']
                       for index, label in enumerate(labels)}
        labels = {label["id"]: index for index, label in enumerate(labels)}

        images = json_data["images"]
        annotations = json_data["annotations"]

        LOG.info(f'{TAG_NAME} {self._name} Number of Images: {len(images)}')
        LOG.info(f'{TAG_NAME} {self._name} Number of Annotations: {len(annotations)}')
        LOG.info(f'{TAG_NAME} {self._name} Categories Values: {labels}')
        LOG.info(f'{TAG_NAME} {self._name} Categories Names: {class_names}')

        records = {
            image['id']: {
                'image_id': image['id'],
                'bboxes': [],
                'labels': [],
                'filepath': os.path.join(img_dir, image['file_name'])
            }
            for image in images
        }

        for ann in annotations:
            bbox_xywh = ann['bbox']
            bbox_xyxy = coco_to_pascal(bbox_xywh)
            records[ann['image_id']]['bboxes'].append(bbox_xyxy)
            records[ann['image_id']]['labels'].append(labels[ann['category_id']])

        return list(records.values()), labels, class_names

    def initalize_augmentations(self, transforms: Dict, img_dir: str):

        transforms_list = []
        for transform_name, transform_kwargs in transforms.items():
            transform_type = TRANSFORMS.get(transform_name)

            if ((transform_type == ResizeTo) or (transform_type == ResizeAugV1)
               or (transform_type == ResizeAugV2)):
                # If the augmentation is resize, then change the out_width and out_height to model resolution
                transform_kwargs["out_width"] = self.model_resolution[1]
                transform_kwargs["out_height"] = self.model_resolution[0]
            elif ((transform_type == BasicMosaicAugmentation) or (transform_type == RoIMosaicAugmentation)):
                # If mosaic check if the path is present in config, else provice
                ann_filepath = transform_kwargs.get("ann_file_path", None)
                data_dir = transform_kwargs.get("data_dir", None)
                if ann_filepath is None:
                    transform_kwargs["ann_file_path"] = self._json_path
                if data_dir is None:
                    transform_kwargs["data_dir"] = img_dir
                # set the model resolution
                transform_kwargs["model_resolution"] = self.model_resolution

            transforms_list.append(transform_type(**transform_kwargs))

        self.augmentations = Compose(transforms_list)

    def generate_groundtruth_data(self, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        heatmap = np.zeros(
            (self.heat_h, self.heat_w, self.num_classes), dtype=np.float32)
        size = np.zeros((self.max_objects, 2), dtype=np.float32)
        offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices = np.zeros((self.max_objects), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects), dtype=np.float32)

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                              dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(heatmap[:, :, int(label)], ct_int, radius)
                size[idx] = 1. * w, 1. * h
                indices[idx] = ct_int[1] * self.heat_w + ct_int[0]
                offset[idx] = ct - ct_int
                reg_mask[idx] = 1.0
        return heatmap, size, indices, offset, reg_mask

    def get_single_data(self, index):
        record = self.records[index]
        bboxes = np.array(record['bboxes'], dtype=np.float32).reshape(-1, 4)
        labels = np.array(record['labels'], dtype=np.int32).reshape(-1, 1)
        filepath = record['filepath']

        assert len(bboxes) == len(
            labels), f'Expects same length for bboxes({len(bboxes)}) and labels({len(labels)})'
        assert os.path.exists(
            filepath), f'Image path {filepath} doesn\'t exists, Please check the path'

        image = cv2.imread(filepath)[:, :, :3]  # Only 3 Channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        org_h, org_w = image.shape[:2]

        # Apply Transforms
        image, bboxes, labels = self.augmentations(image, bboxes, labels)
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)

        # Convert to Float Image
        image = image.astype(np.float32)

        # Select only self.max_objects
        bboxes = bboxes[:self.max_objects, :].astype(np.float32)
        labels = labels[:self.max_objects, :].astype(np.int32)

        # Downscale as per Centernet
        bboxes /= self.down_ratio

        # Get the ground truth data
        heatmap, size, indices, offset, reg_mask = self.generate_groundtruth_data(
            bboxes, labels)

        # Calculate the number of zeros to pad
        num_rows_to_pad = self.max_objects - len(bboxes)

        # Pad zeros to bboxes
        bboxes = np.pad(bboxes, ((0, num_rows_to_pad), (0, 0)), 'constant')
        # Pad zeros to labels
        labels = np.pad(labels, ((0, num_rows_to_pad), (0, 0)), 'constant')

        assert len(bboxes) == len(labels) == len(size) == len(indices) == len(offset) == len(reg_mask), \
            f'Expected same length for bboxes({len(bboxes)}), labels({len(labels)}), size({len(size)}) ' \
            f'indices({len(indices)}), reg({len(offset)}), reg_mask({len(reg_mask)})'

        scale_h = float(self.model_resolution[0]) / float(org_h)
        scale_w = float(self.model_resolution[1]) / float(org_w)

        data = {
            'input': image,
            'bboxes': bboxes,
            'labels': labels,
            'heatmap': heatmap,
            'size': size,
            'indices': indices,
            'offset': offset,
            'reg_mask': reg_mask,
            'filepath': filepath,
            'scale_h': scale_h,
            'scale_w': scale_w,
            'image_id': int(record['image_id'])
        }

        return data

    @property
    def output_names(self):
        names = [
            'input',
            'bboxes',
            'labels',
            'heatmap',
            'size',
            'indices',
            'offset',
            'reg_mask',
            'filepath',
            'scale_h',
            'scale_w',
            'image_id'
        ]
        return names

    @property
    def output_types(self) -> Dict[str, tf.DType]:
        types = {
            'input': tf.float32,
            'bboxes': tf.float32,
            'labels': tf.int32,
            'heatmap': tf.float32,
            'size': tf.float32,
            'indices': tf.int64,
            'offset': tf.float32,
            'reg_mask': tf.float32,
            'filepath': tf.string,
            'scale_h': tf.float32,
            'scale_w': tf.float32,
            'image_id': tf.int32
        }
        return types

    @property
    def output_shapes(self) -> Dict[str, tf.TensorShape]:
        shapes = {
            'input': tf.TensorShape([self.out_h, self.out_w, 3]),
            'bboxes': tf.TensorShape([self.max_objects, 4]),
            'labels': tf.TensorShape([self.max_objects, 1]),
            'heatmap': tf.TensorShape([self.heat_h, self.heat_w, self.num_classes]),
            'size': tf.TensorShape([self.max_objects, 2]),
            'indices': tf.TensorShape([self.max_objects]),
            'offset': tf.TensorShape([self.max_objects, 2]),
            'reg_mask': tf.TensorShape([self.max_objects]),
            'filepath': tf.TensorShape([]),
            'scale_h': tf.TensorShape([]),
            'scale_w': tf.TensorShape([]),
            'image_id': tf.TensorShape([])
        }
        return shapes

    def scheduler(self, epoch):
        if epoch < 90:
            return self.init_lr
        elif epoch < 120: 
            return self.init_lr * 0.1
        else:
            return self.init_lr * 0.01
