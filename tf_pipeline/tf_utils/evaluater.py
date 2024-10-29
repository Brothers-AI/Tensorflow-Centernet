import os
import sys
import time
import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import BatchDataset

from tf_pipeline.utils.registry.datasets import DATASETS
from tf_pipeline.datasets import DefaultDataset
from tf_pipeline.utils.configs.config import Config, ValParams
from tf_pipeline.callbacks.map_callback import PredNValidateCenterNetMAPCallback, PredNValidateCenterNetCOCOCallback
from tf_pipeline.models.create import create_tf_model


class TensorflowEvaluater(object):
    def __init__(self, config_path: str):

        config = Config.parse_file(config_path)

        # Logging config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger()

        self.logger.info(f"Config file -> {config_path}")
        self.logger.info(f"Config file contents\n{config}")

        # Create val dataset
        self.val_params, self.val_dataset, self.val_ds, self.num_val_iters = self.initalize_val_dataset(
            config)

        # Create and load the model
        self.model = self.create_and_load_model(config)

        # Initalize callbacks
        self.callbacks = self.initalize_callbacks(config)

    def initalize_val_dataset(self, config: Config) -> Tuple[ValParams, DefaultDataset, BatchDataset]:

        # Get the val params
        val_params = config.val_params
        val_dataset_name, val_dataset_kwargs = val_params.dataset_config()

        # Get the transforms
        val_transforms = val_params.transforms

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

    def initalize_callbacks(self, config: Config):
        # Map Callback
        map_cb = PredNValidateCenterNetCOCOCallback(self.model, self.val_ds, self.val_dataset.num_classes,
                                                    config, self.val_dataset.val_json_path, save_outputs=False,
                                                    output_dir="validation_outputs")

        callbacks = [map_cb]
        return callbacks

    def create_and_load_model(self, config: Config):
        # Create the TF Model
        model = create_tf_model(config.tf_module, config.model_config, [
            *config.model_resolution, 3], num_classes=self.val_dataset.num_classes)
        if config.ckpt_path:
            self.logger.info(
                f"Loading from checkpoint {config.ckpt_path}")
            model.load_weights(config.ckpt_path).expect_partial()
        else:
            self.logger.error(
                f"Checkpoint path not given, Please give the checkpoint path")
            exit(-1)
        model.compile()
        model.summary()
        return model

    def evaluate(self):
        self.model.evaluate(
            self.val_ds,
            steps=self.num_val_iters,
            batch_size=self.val_params.batch_size,
            verbose=1,
            callbacks=self.callbacks,
            workers=self.val_params.num_workers,
            use_multiprocessing=True)
        return
