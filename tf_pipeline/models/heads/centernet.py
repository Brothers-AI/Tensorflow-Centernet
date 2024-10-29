import logging

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import HEADS

LOG = logging.getLogger()
TAG_NAME = "[Head]"


@HEADS.register_module()
class CenterNet(layers.Layer):
    def __init__(self, interm_channels: int, num_classes: int):
        super(CenterNet, self).__init__(name='CenterNet')

        self.head_name = "[Centernet]"
        LOG.info(f"{TAG_NAME} {self.head_name}")
        LOG.info(
            f"{TAG_NAME} {self.head_name} interm_channels: {interm_channels}")
        LOG.info(f"{TAG_NAME} {self.head_name} num_classes: {num_classes}")

        self.heatmap = keras.Sequential([
            layers.Conv2D(interm_channels, kernel_size=3, padding='same',
                          use_bias=False, kernel_initializer=keras.initializers.RandomNormal(0.01),
                          kernel_regularizer=keras.regularizers.l2(1.25e-5)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(num_classes, 1, padding='valid',
                          kernel_initializer=keras.initializers.RandomNormal(
                              0.01),
                          kernel_regularizer=keras.regularizers.l2(1.25e-5),
                          bias_initializer=tf.constant_initializer(
                              -np.log((1.0 - 0.1) / 0.1)),
                          activation=tf.nn.sigmoid)],
                          name='heatmap')

        self.bbox_size = keras.Sequential([
            layers.Conv2D(interm_channels, kernel_size=3, padding='same',
                          use_bias=False, kernel_initializer=keras.initializers.RandomNormal(0.01),
                          kernel_regularizer=keras.regularizers.l2(1.25e-5)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(2, 1, padding='valid', activation=None,
                          kernel_initializer=keras.initializers.RandomNormal(
                              0.01),
                          kernel_regularizer=keras.regularizers.l2(1.25e-5))],
                          name='bounding_box_size')

        self.local_offset = keras.Sequential([
            layers.Conv2D(interm_channels, kernel_size=3, padding='same',
                          use_bias=False, kernel_initializer=keras.initializers.RandomNormal(0.01),
                          kernel_regularizer=keras.regularizers.l2(1.25e-5)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(2, 1, padding='valid', activation=None,
                          kernel_initializer='he_uniform',
                          kernel_regularizer=keras.regularizers.l2(1.25e-5))],
                          name='local_offset')

    def build(self, input_shape):
        with tf.name_scope('heatmap'):
            self.heatmap.build(input_shape)
        with tf.name_scope('bounding_box_size'):
            self.bbox_size.build(input_shape)
        with tf.name_scope('local_offset'):
            self.local_offset.build(input_shape)

    def call(self, input_tensors, training=False):
        heatmap_out = self.heatmap(input_tensors, training=training)
        bbox_size_out = self.bbox_size(input_tensors, training=training)
        local_offset_out = self.local_offset(input_tensors, training=training)

        return [heatmap_out, bbox_size_out, local_offset_out]
