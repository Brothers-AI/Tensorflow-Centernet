import os
import json
import logging
from typing import Union, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import BACKBONES

LOG = logging.getLogger()
TAG_NAME = "[Backbone]"

"""
# layer 5 = 5
    {
        [104-216, k=5, s=2, e=6]
        [216-216, k=5, s=1, e=3] + input
        [216-216, k=5, s=1, e=3] + input
        [216-216, k=3, s=1, e=3] + input
        [216-360, k=5, s=1, e=6]
    }
# layer 4 = 8
    {
        [48-88, k=3, s=2, e=6]
        [88-88, k=3, s=1, e=3] + input
        [88-104, k=5, s=1, e=6]
        [104-104, k=3, s=1, e=3] + input
        [104-104, k=3, s=1, e=3] + input
        [104-104, k=3, s=1, e=3] + input
    }
# layer 3 = 4
    {
        [32-48, k=3, s=2, e=6]
        [48-48, k=3, s=1, e=3] + input
        [48-48, k=3, s=1, e=3] + input
        [48-48, k=5, s=1, e=3] + input
    }
# layer 2 = 4
    {
        [24-32, k=3, s=2, e=6]
        [32-32, k=3, s=1, e=3] + input
        [32-32, k=3, s=1, e=3] + input
        [32-32, k=3, s=1, e=3] + input
    }
# layer 1 = 1 [40-24] + stem[3-40]
    {
        # stem
        [Conv, [3-40, s=2]]
        # Layer
        [40-24, k=3, s=1, e=1]
    }
"""

class Identity(keras.layers.Layer):
    def __init__(self, name: str):
        super(Identity, self).__init__(name=name)

    def call(self, inputs, training=False):
        return inputs


class MBInvertedConv(layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expand_ratio: float = 6):
        super(MBInvertedConv, self).__init__(name="MBInvertedConv")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if expand_ratio > 1:
            feature_dim = round(in_channels * expand_ratio)
            self.inverted_bottleneck = keras.Sequential([
                layers.Conv2D(feature_dim, kernel_size=1,
                                strides=1, padding='valid', use_bias=False),
                layers.BatchNormalization()])
            self.use_inverted_block = True
        else:
            self.inverted_bottleneck = Identity('mb_identity')
            self.use_inverted_block = False

        self.depth_conv = keras.Sequential([
            layers.DepthwiseConv2D(
                self.kernel_size, self.stride, padding='same'),
            layers.BatchNormalization(),])
        self.point_linear = keras.Sequential([
            layers.Conv2D(self.out_channels, 1, 1, padding='same'),
            layers.BatchNormalization()])

    def call(self, inputs, training=False):
        if self.use_inverted_block:
            inverted_bottleneck_out = tf.nn.relu6(self.inverted_bottleneck(
                inputs, training=training))
        else:
            inverted_bottleneck_out = inputs

        depth_conv_out = tf.nn.relu6(self.depth_conv(
            inverted_bottleneck_out, training=training))
        point_out = self.point_linear(
            depth_conv_out, training=training)
        return point_out


class StemBlock(layers.Layer):
    def __init__(self, name: str, interm_channels: int, out_channels: int):
        super(StemBlock, self).__init__(name=name)

        self.stem_basic = keras.Sequential([
            layers.Conv2D(interm_channels, kernel_size=3, strides=2,
                          padding='same', use_bias=False),
            layers.BatchNormalization(),
        ])

        self.branch = MBInvertedConv(interm_channels, out_channels, 3, stride=1,
                                     expand_ratio=1)
    
    def call(self, inputs, training=False):
        stem_out = tf.nn.relu6(self.stem_basic(inputs, training=training))
        branch_out = self.branch(stem_out, training=training)
        return branch_out

class TransitionLayer(layers.Layer):
    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionLayer, self).__init__()

        self.layer = MBInvertedConv(in_channels, out_channels, kernel_size=3,
                                    stride=2, expand_ratio=6)

    def call(self, input_tensor, training=False):
        layer_out = self.layer(input_tensor, training=training)
        return layer_out

class LayerBlock(layers.Layer):
    def __init__(self, name: str, in_channels: Sequence[int], out_channels: Sequence[int],
                 kernel_sizes: Sequence[int], strides: Sequence[int], expand_ratios: Sequence[int],
                 add_shortcuts: Sequence[bool]):
        super(LayerBlock, self).__init__(name=name)

        assert len(in_channels) == len(out_channels), f"Expected same length for in_channels \
                                                        ({len(in_channels)}) and out_channels ({len(out_channels)})"

        assert len(kernel_sizes) == len(strides) == len(expand_ratios) == len(add_shortcuts), f"Expected same length for all the inputs in LayerBlock"

        self.transition_layer = MBInvertedConv(in_channels[0], out_channels[0],
                                               kernel_size=kernel_sizes[0],
                                               stride=strides[0],
                                               expand_ratio=expand_ratios[0])

        self.add_shortcuts = add_shortcuts

        self.dense_layers = []
        for idx in range(1, len(in_channels)):
            layer = MBInvertedConv(in_channels[idx], out_channels[idx],
                                   kernel_sizes[idx], strides[idx],
                                   expand_ratios[idx])
            self.dense_layers.append(layer)
    
    def call(self, inputs, training=False):
        prev_out = self.transition_layer(inputs, training=training)

        if self.add_shortcuts[0]:
            prev_out = tf.add(inputs, prev_out)

        for idx, layer in enumerate(self.dense_layers):
            out = layer(prev_out, training=training)
            if self.add_shortcuts[idx + 1]:
                prev_out = tf.add(prev_out, out)
            else:
                prev_out = out
        return prev_out


@BACKBONES.register_module()
class ProxylessNAS(layers.Layer):
    def __init__(self, config_path: str):
        super(ProxylessNAS, self).__init__(name='ProxylessNAS')

        assert os.path.exists(config_path), f"ProxylessNAS config_path file {config_path}, doesn't found. Please check"

        with open(config_path, "r") as fp:
            config_data = json.load(fp)

        self.backbone_name = "[ProxylessNAS]"

        # Stem Block
        stem_config = config_data['stem_block']
        self.stem_block = StemBlock('stem_block', stem_config['interm_channels'], stem_config['out_channels'])

        # Layers Config
        layers_config = config_data['layers']

        layers_in_channels = []
        layers_out_channels = []
        layers_kernel_sizes = []
        layers_strides = []
        layers_exp_ratios = []
        layers_add_shortcuts = []

        for layer in layers_config:
            layers_in_channels.append(layer['in_channels'])
            layers_out_channels.append(layer['out_channels'])
            layers_kernel_sizes.append(layer['kernel_sizes'])
            layers_strides.append(layer['strides'])
            layers_exp_ratios.append(layer['expand_ratios'])
            layers_add_shortcuts.append(layer['add_shortcuts'])

        self.num_layers = len(layers_config)

        self.layer_name = "layer_{i}"

        for idx in range(self.num_layers):
            layer = LayerBlock(f'block_{idx}', layers_in_channels[idx],
                               layers_out_channels[idx], layers_kernel_sizes[idx],
                               layers_strides[idx], layers_exp_ratios[idx],
                               layers_add_shortcuts[idx])
            self.__setattr__(self.layer_name.format(i=idx), layer)

    def call(self, input_tensor, training=False):
        x = self.stem_block(input_tensor, training=training)
        outputs = [x]
        for i in range(self.num_layers):
            x = self.__getattribute__(
                self.layer_name.format(i=i))(x, training=training)
            outputs.append(x)
        return outputs


if __name__ == "__main__":
    data = tf.random.uniform([1, 512, 768, 3])

    proxyless_nas_model = ProxylessNAS()
    model_out = proxyless_nas_model(data)
    for out in model_out:
        print(out.shape)
