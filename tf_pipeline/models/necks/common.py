import logging
from typing import Union, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import NECKS

LOG = logging.getLogger()
TAG_NAME = "[Neck]"


@NECKS.register_module()
class Upsample2D(layers.Layer):
    def __init__(self, scale_factor: int, mode: str = 'nearest', idx_to_take: int = 0):
        super(Upsample2D, self).__init__(name='upsample_2d')

        self.neck_name = "[Upsample2D]"
        LOG.info(f"{TAG_NAME} {self.neck_name} mode: {mode}")
        LOG.info(f"{TAG_NAME} {self.neck_name} scale_factor: {scale_factor}")
        LOG.info(f"{TAG_NAME} {self.neck_name} idx_to_take: {idx_to_take}")

        self.mode = mode
        self.scale_factor = scale_factor
        self.idx_to_take = idx_to_take

    def call(self, inputs, training=False):
        if not isinstance(inputs, Sequence):
            raise TypeError(
                f"Expected inputs to be of type Sequence, but found {type(inputs)}")

        input_to_resize = inputs[self.idx_to_take]
        in_h, in_w = input_to_resize.shape[1:3]
        resize_out = tf.compat.v1.image.resize(input_to_resize,
                                               size=[in_h * self.scale_factor,
                                                     in_w * self.scale_factor],
                                               method=self.mode,
                                               align_corners=False if self.mode == 'nearest' else True)
        return resize_out


@NECKS.register_module()
class UpsampleConv(layers.Layer):
    def __init__(self, out_channels: int, idx_to_take: int = 0):
        super(UpsampleConv, self).__init__(name='upsample_conv')

        self.neck_name = "[UpsampleConv]"
        LOG.info(f"{TAG_NAME} {self.neck_name} out_channels: {out_channels}")
        LOG.info(f"{TAG_NAME} {self.neck_name} idx_to_take: {idx_to_take}")

        self.idx_to_take = idx_to_take

        self.deconv_layer = keras.Sequential([
            layers.Conv2DTranspose(int(out_channels), kernel_size=4,
                                   strides=2, padding='same', use_bias=False,
                                   kernel_initializer='he_uniform',
                                   kernel_regularizer=keras.regularizers.l2(1.25e-5)),
            layers.BatchNormalization(),
            layers.Activation('relu')])

    def call(self, inputs, training=False):
        if not isinstance(inputs, Sequence):
            raise TypeError(
                f"Expected inputs to be of type Sequence, but found {type(inputs)}")
        return self.deconv_layer(inputs[self.idx_to_take], training=training)
