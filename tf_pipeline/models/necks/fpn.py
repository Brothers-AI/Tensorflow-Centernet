import logging
from typing import Union, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import NECKS

LOG = logging.getLogger()
TAG_NAME = "[Neck]"


@NECKS.register_module()
class FPN(layers.Layer):
    def __init__(self, num_inputs: int, out_channels: int, num_outs: int,
                 start_level: int = 0, end_level: int = -1,
                 upsample_mode: str = "nearest",
                 add_extra_convs: bool = False,
                 activation_before_extra_convs: bool = False):
        super(FPN, self).__init__(name="FPN")

        self.neck_name = "[FPN]"

        LOG.info(f"{TAG_NAME} {self.neck_name} num_inputs: {num_inputs}")
        LOG.info(f"{TAG_NAME} {self.neck_name} out_channels: {out_channels}")
        LOG.info(f"{TAG_NAME} {self.neck_name} num_outs: {num_outs}")
        LOG.info(f"{TAG_NAME} {self.neck_name} start_level: {start_level}")
        LOG.info(f"{TAG_NAME} {self.neck_name} end_level: {end_level}")
        LOG.info(f"{TAG_NAME} {self.neck_name} upsample_mode: {upsample_mode}")
        LOG.info(
            f"{TAG_NAME} {self.neck_name} add_extra_convs: {add_extra_convs}")
        LOG.info(
            f"{TAG_NAME} {self.neck_name} activation_before_extra_convs: {activation_before_extra_convs}")

        self.out_channels = int(out_channels)
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outs)
        self.upsample_mode = upsample_mode

        # [Bool, 'on_input', 'on_output', 'on_lateral']
        self.add_extra_convs = add_extra_convs
        self.activation_before_extra_convs = activation_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_inputs
            assert num_outs >= self.num_inputs - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= self.num_inputs
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = []
        self.fpn_convs = []

        for i in range(self.start_level, self.backbone_end_level):

            # Lateral Convolutions
            l_conv = keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=1, padding='same'),
                layers.BatchNormalization(),
                layers.Activation('relu')])

            fpn_conv = keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=3, padding='same'),
                layers.BatchNormalization(),
                layers.Activation('relu')])

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # Extra Convolutions
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                extra_fpn_conv = keras.Sequential([
                    layers.Conv2D(out_channels, kernel_size=3,
                                  strides=2, padding='same'),
                    layers.BatchNormalization(),
                    layers.Activation('relu')])
                self.fpn_convs.append(extra_fpn_conv)

    def call(self, inputs, training=False):
        if not isinstance(inputs, Sequence):
            raise TypeError(
                f"Expected inputs to be of type Sequence, but found {type(inputs)}")

        if len(inputs) != self.num_inputs:
            raise AssertionError(
                f"Expected {self.num_inputs} inputs, but found {len(inputs)}")

        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level], training=training)
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build Top-Down Path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            in_h, in_w = laterals[i].shape[1:3]
            resize_out = tf.compat.v1.image.resize(laterals[i],
                                                   size=[in_h * 2, in_w * 2],
                                                   method=self.upsample_mode,
                                                   align_corners=False if self.upsample_mode == 'nearest' else True)
            laterals[i - 1] = tf.add(laterals[i - 1], resize_out)

        # Build Outputs
        outs = [
            self.fpn_convs[i](laterals[i], training=training)
            for i in range(used_backbone_levels)
        ]

        # Extra convolutions
        if self.num_outputs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outputs - used_backbone_levels):
                    outs.append(tf.nn.max_pool2d(
                        outs[-1], 1, strides=2, padding='SAME'))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError(
                        f"Not implemented for add_extra_convs on \"{self.add_extra_convs}\"")

                outs.append(self.fpn_convs[used_backbone_levels](
                    extra_source, training=training))

                for i in range(used_backbone_levels + 1, self.num_outputs):
                    if self.activation_before_extra_convs:
                        outs.append(self.fpn_convs[i](
                            tf.nn.relu(outs[-1])), training=training)
                    else:
                        outs.append(self.fpn_convs[i](
                            outs[-1]), training=training)

        return tuple(outs)
