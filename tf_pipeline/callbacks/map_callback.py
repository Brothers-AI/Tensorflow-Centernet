import os
from typing import Dict, Union
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import BatchDataset

from tf_pipeline.utils.configs.config import Config
from tf_pipeline.models.evaluation.centernet.mean_average_precision import MAP
from tf_pipeline.models.evaluation.centernet.coco_evaluation import COCOEvaluation
from tf_pipeline.models.post_proc.centernet import CenterNetPostProc
from tf_pipeline.utils.registry.post_proc import POST_PROC
from tf_pipeline.utils.visualization import overlap_bbox_onto_image


class PredNValidateCenterNetCOCOCallback(tf.keras.callbacks.Callback):
    def __init__(self, model: tf.keras.Model, val_ds: BatchDataset, num_classes: int, config: Config,
                 val_json_path: str, save_outputs: bool = False,
                 output_dir: str = "validation_outputs"):
        super(PredNValidateCenterNetCOCOCallback, self).__init__()

        assert os.path.exists(
            val_json_path), f"Json path {val_json_path} doesn't found. Please check"

        if not isinstance(config, Config):
            raise TypeError(
                f"Expected config of type {type(Config)}, but found {type(config)}")

        self.coco_eval = COCOEvaluation(val_json_path)

        # Create post processing module
        post_proc_name, post_proc_kwargs = config.model_config.postproc_config()
        self.post_proc_module: CenterNetPostProc = POST_PROC.get(
            post_proc_name)(**post_proc_kwargs)
        score_threshold = float(post_proc_kwargs.get("score_threshold", 0.3))
        iou_threshold = float(post_proc_kwargs.get("iou_threshold", 0.5))

        # Module to predict the data
        self.model = model

        # Create dataset iter
        self.val_dataset_iter = iter(val_ds)

        # Save outputs if enabled
        self.save_outputs = save_outputs
        if save_outputs:
            # Color maps
            self.color_maps = np.random.randint(
                0, 255, size=(num_classes, 3)).astype(np.uint32)

            # Create output directory
            self.save_out_dir = output_dir
            os.makedirs(self.save_out_dir, exist_ok=True)

    def on_test_batch_begin(self, batch, logs=None):
        # Reset the coco
        self.coco_eval.reset_data()
        return

    def on_test_batch_end(self, batch, logs=None):
        # Get the batch data from dataset
        batch_data = next(self.val_dataset_iter)

        # Prediction from the model
        model_preds = self.model.predict_on_batch(batch_data)

        # Decode and filter detections
        batch_detections = self.post_proc_module(model_preds)

        for image, detections, filepath, scale_h, scale_w, image_id in zip(batch_data['input'], batch_detections,
                                                                 batch_data['filepath'], batch_data['scale_h'],
                                                                 batch_data['scale_w'], batch_data['image_id']):

            # overlay and Save
            if self.save_outputs:
                overlayed_image = overlap_bbox_onto_image(
                    image.numpy(
                    ), detections[:, :4], detections[:, -1], self.color_maps,
                    renormalize=True, mean=[127, 127, 127], std=[127, 127, 127])

                output_path = os.path.join(
                    self.save_out_dir, os.path.basename(filepath.numpy().decode()))
                cv2.imwrite(output_path, overlayed_image[:, :, ::-1])

            scaled_detections = np.array(detections, dtype=np.float32)
            scaled_detections[:, [0, 2]] /= scale_w
            scaled_detections[:, [1, 3]] /= scale_h

            self.coco_eval.update_state(scaled_detections, image_id)
        return
    
    def on_test_end(self, logs=None):
        self.coco_eval.result()
        self.coco_eval.reset_data()
        return
