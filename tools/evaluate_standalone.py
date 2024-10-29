from typing import Dict, List, Tuple
import os
import sys
import argparse
import logging
import json

import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import BatchDataset

sys.path.append(".")

from tf_pipeline.models.create import create_tf_model
from tf_pipeline.utils.configs.config import Config, ValParams
from tf_pipeline.datasets import DefaultDataset, DefaultDatasetV2
from tf_pipeline.models.evaluation.centernet.mean_average_precision import MAP
from tf_pipeline.models.evaluation.centernet.coco_evaluation import COCOEvaluation
from tf_pipeline.models.evaluation.custom import CustomEvaluation
from tf_pipeline.models.post_proc.centernet import CenterNetPostProc
from tf_pipeline.utils.visualization import overlap_bbox_onto_image
from tf_pipeline.utils.registry.datasets import DATASETS
from tf_pipeline.utils.registry.post_proc import POST_PROC


def parser_args():
    parser = argparse.ArgumentParser("Evaluation")
    parser.add_argument("--config", required=True,
                        help="Path of the config file")
    parser.add_argument("--eval_type", choices=['coco', 'pascal', 'custom'], default='coco',
                        help='Type of evaluation')
    parser.add_argument("--save_outputs", action='store_true',
                        help='To store the outputs')
    parser.add_argument("--save_out_dir", default="val_outputs", type=str,
                        help='Save directory to save the results')
    return parser.parse_args()


def initalize_val_dataset(config: Config) -> Tuple[ValParams, DefaultDataset, BatchDataset, int]:
    # Get the val params
    val_params = config.val_params
    val_dataset_name, val_dataset_kwargs = val_params.dataset_config()

    # Get the transforms
    val_transforms = val_params.transforms

    if 'V2' in val_dataset_name:
        # Create the val dataset
        val_dataset_kwargs['batch_size'] = 1
        val_dataset_kwargs['shuffle'] = False
        val_dataset: DefaultDatasetV2 = DATASETS.get(val_dataset_name)(model_resolution=config.model_resolution,
                                                                        split="val", transforms=val_transforms,
                                                                        **val_dataset_kwargs)
        
        val_ds = val_dataset
        num_iters = len(val_dataset)
    else:
        # Create the val dataset
        val_dataset: DefaultDataset = DATASETS.get(val_dataset_name)(model_resolution=config.model_resolution,
                                                                    split="val", transforms=val_transforms,
                                                                    **val_dataset_kwargs)
        val_ds = tf.data.Dataset.from_generator(val_dataset, output_types=val_dataset.output_types,
                                                output_shapes=val_dataset.output_shapes)
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        val_ds = val_ds.with_options(options)
        val_ds = val_ds.batch(val_params.batch_size, drop_remainder=True)

        num_iters = int(len(val_dataset) // val_params.batch_size)

    return val_params, val_dataset, val_ds, num_iters


def create_and_load_model(config: Config, logger: logging.Logger, num_classes: int):
    # Create the TF Model
    model = create_tf_model(config.tf_module, config.model_config, [
        *config.model_resolution, 3], num_classes=num_classes)
    if config.ckpt_path:
        logger.info(
            f"Loading from checkpoint {config.ckpt_path}")
        model.load_weights(config.ckpt_path).expect_partial()
    else:
        logger.error(
            f"Checkpoint path not given, Please give the checkpoint path")
        exit(-1)
    model.compile()
    model.summary()
    return model


def main():

    args = parser_args()

    config_path = args.config
    eval_type = args.eval_type
    save_outputs = args.save_outputs
    save_out_dir = args.save_out_dir

    # get config
    config = Config.parse_file(config_path)

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger()

    logger.info(f"Config file -> {config_path}")
    logger.info(f"Config file contents\n{config}")

    # Create val dataset
    val_params, val_dataset, val_ds, num_val_iters = initalize_val_dataset(config)

    # Create and load model
    model = create_and_load_model(config, logger, val_dataset.num_classes)

    # Create post processing module
    post_proc_name, post_proc_kwargs = config.model_config.postproc_config()
    post_proc_module: CenterNetPostProc = POST_PROC.get(
        post_proc_name)(**post_proc_kwargs)
    score_threshold = float(post_proc_kwargs.get("score_threshold", 0.3))
    iou_threshold = float(post_proc_kwargs.get("iou_threshold", 0.5))

    if eval_type == "coco":
        coco_eval = COCOEvaluation(val_dataset.val_json_path)
    elif eval_type == "pascal":
        pascal_map = MAP(val_dataset.num_classes, iou_threshold, score_threshold)
    elif eval_type == 'custom':
        custom_eval = CustomEvaluation(val_dataset.val_json_path, save_dir=save_out_dir)
    else:
        raise NotImplementedError(f"Not implemented for type {eval_type}")
    
    if save_outputs:
        # Color maps
        color_maps = np.random.randint(
            0, 255, size=(val_dataset.num_classes, 3)).astype(np.uint32)

        # Create output directory
        os.makedirs(save_out_dir, exist_ok=True)
    
    for batch_data in tqdm(val_ds, desc=f"Evalauting: type: {eval_type}", total=num_val_iters):

        # predictions from model
        model_preds = model.predict_on_batch(batch_data)

        # decode and filter detections
        batch_detections = post_proc_module(model_preds)

        for image, gt_bboxes, gt_labels, gt_masks, detections, filepath, scale_h, scale_w, image_id in zip(batch_data['input'],
                                                                                                 batch_data['bboxes'], batch_data['labels'], batch_data['reg_mask'],
                                                                                                 batch_detections, batch_data['filepath'], batch_data['scale_h'],
                                                                                                 batch_data['scale_w'], batch_data['image_id']):
            # overlay and Save
            if save_outputs:
                if isinstance(image, tf.Tensor):
                    overlayed_image = overlap_bbox_onto_image(
                        image.numpy(
                        ), detections[:, :4], detections[:, -1], color_maps,
                        renormalize=True, mean=[127, 127, 127], std=[127, 127, 127])
                    output_path = os.path.join(
                        save_out_dir, os.path.basename(filepath.numpy().decode()))
                else:
                    overlayed_image = overlap_bbox_onto_image(
                        image, detections[:, :4], detections[:, -1], color_maps,
                        renormalize=True, mean=[127, 127, 127], std=[127, 127, 127])

                    output_path = os.path.join(
                        save_out_dir, os.path.basename(filepath))
                cv2.imwrite(output_path, overlayed_image[:, :, ::-1])

            scaled_detections = np.array(detections, dtype=np.float32)
            scaled_detections[:, [0, 2]] /= scale_w
            scaled_detections[:, [1, 3]] /= scale_h
            
            if eval_type == 'coco':
                coco_eval.update_state(scaled_detections, image_id)
            elif eval_type == 'pascal':
                if isinstance(gt_bboxes, tf.Tensor):
                    mask = gt_masks.numpy().astype(np.bool)
                    bboxes = gt_bboxes.numpy() * val_dataset.down_ratio
                    labels = gt_labels.numpy()
                else:
                    mask = gt_masks.astype(np.bool)
                    bboxes = gt_bboxes * val_dataset.down_ratio
                    labels = gt_labels
                bboxes = bboxes[mask]
                labels = labels[mask]
                bboxes[:, [0, 2]] /= scale_w
                bboxes[:, [1, 3]] /= scale_h
                pascal_map.update_state(scaled_detections, bboxes, labels)
            elif eval_type == 'custom':
                if isinstance(gt_bboxes, tf.Tensor):
                    bboxes = gt_bboxes.numpy() * val_dataset.down_ratio
                    labels = gt_labels.numpy()
                else:
                    bboxes = gt_bboxes * val_dataset.down_ratio
                    labels = gt_labels
                bboxes[:, [0, 2]] /= scale_w
                bboxes[:, [1, 3]] /= scale_h
                custom_eval.update_state(scaled_detections, bboxes, labels)
    
    if eval_type == 'coco':
        coco_eval.result()
        coco_eval.reset_data()
    elif eval_type == 'pascal':
        pascal_result = pascal_map.result()
        logger.info("{}".format(json.dumps(pascal_result, indent=4)))
        pascal_map.reset_states()
    elif eval_type == 'custom':
        custom_eval.result()
        custom_eval.reset_data()
    return


if __name__ == "__main__":
    main()
