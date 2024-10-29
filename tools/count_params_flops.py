from typing import Dict, List, Tuple
import os
import sys
import argparse
import logging

import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import BatchDataset

sys.path.append(".")

from tf_pipeline.models.create import create_tf_model
from tf_pipeline.utils.configs.config import Config, ValParams
from tf_pipeline.datasets import DefaultDataset
from tf_pipeline.models.evaluation.centernet.mean_average_precision import MAP
from tf_pipeline.models.evaluation.centernet.coco_evaluation import COCOEvaluation
from tf_pipeline.models.post_proc.centernet import CenterNetPostProc
from tf_pipeline.utils.visualization import overlap_bbox_onto_image
from tf_pipeline.utils.registry.datasets import DATASETS
from tf_pipeline.utils.registry.post_proc import POST_PROC

def parse_args():
    parser = argparse.ArgumentParser("Count params and flops")
    parser.add_argument("--config", required=True, help="Path of the config file")
    parser.add_argument("--batch_size", type=int, default=1, help='Batch size of input to profile')
    parser.add_argument("--input_size", type=str, default='512,768',
                        help='Input size of the model to profile (H, W)')
    parser.add_argument("--num_classes", type=int, default=7,
                        help='Num classes in the head')
    return parser.parse_args()

def get_flops(model: tf.keras.Model, batch_size: int, input_size: List[int]):
    # Taken from https://github.com/tensorflow/tensorflow/issues/32809

    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    # Compute FLOPs for one sample
    inputs = [tf.TensorSpec([batch_size] + input_size, dtype=tf.float32)]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2

def main():

    args = parse_args()
    config_file = args.config
    input_size = args.input_size
    batch_size = args.batch_size

    input_size = list(map(int, input_size.split(",")))

    config = Config.parse_file(config_file)

    # Create the TF Model
    model = create_tf_model(config.tf_module, config.model_config,
                            [*input_size, 3], num_classes=args.num_classes)
    model.compile()
    model.summary()

    g_flops = get_flops(model, batch_size, [*input_size, 3])
    print(f"GFlops: {g_flops}")

    return

if __name__ == "__main__":
    main()