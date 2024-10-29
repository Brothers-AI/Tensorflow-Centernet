from typing import Dict, Any
import logging

import tensorflow as tf

from tf_pipeline.utils.registry.models import BACKBONES, NECKS, HEADS

LOG = logging.getLogger()
TAG_NAME = "[BUILD]"


def build_backbone(name: str, kwargs: Dict[str, Any]) -> tf.Module:
    LOG.info(f"{TAG_NAME} build_backbone: name -> {name}")
    backbone_module: tf.Module = BACKBONES.get(name)(**kwargs)
    return backbone_module


def build_neck(name: str, kwargs: Dict[str, Any]) -> tf.Module:
    LOG.info(f"{TAG_NAME} build_neck: name -> {name}")
    neck_module: tf.Module = NECKS.get(name)(**kwargs)
    return neck_module


def build_head(name: str, kwargs: Dict[str, Any]) -> tf.Module:
    LOG.info(f"{TAG_NAME} build_head: name -> {name}")
    head_module: tf.Module = HEADS.get(name)(**kwargs)
    return head_module
