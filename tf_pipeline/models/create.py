from typing import Sequence
import logging

import tensorflow.keras as keras

from tf_pipeline.models.model import DetectionModel
from tf_pipeline.utils.registry.tf_utils import TF_MODULE
from tf_pipeline.utils.configs.config import ModelConfig

LOG = logging.getLogger()
TAG_NAME = "[CREATE]"

def create_tf_model(tf_module_name: str, config: ModelConfig,
                    input_shape: Sequence[int], num_classes: int, is_v2_dataset: bool = False) -> keras.Model:
    if not isinstance(config, ModelConfig):
        raise TypeError(f"Expected config type to be of ModelConfig, but found of type {type(config)}")

    if len(input_shape) < 3:
        raise TypeError(f"Expected input shape of length 3 (H, W, C), but found of length {len(input_shape)}")

    LOG.info(f"{TAG_NAME} create_tf_model: name -> {tf_module_name}")

    det_model = DetectionModel(config, num_classes)
    input_layer = keras.Input(shape=input_shape, name='input')
    tf_module = TF_MODULE.get(tf_module_name)
    return tf_module(inputs=[input_layer], outputs=det_model.call(input_layer), config=config, is_v2_dataset=is_v2_dataset)
