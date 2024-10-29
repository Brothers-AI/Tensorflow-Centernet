import logging
from typing import Union, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import BACKBONES

LOG = logging.getLogger()
TAG_NAME = "[Backbone]"

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
                layers.BatchNormalization(),
                layers.Activation(tf.nn.relu6)])
            self.use_inverted_block = True
        else:
            self.inverted_bottleneck = Identity('mb_identity')
            self.use_inverted_block = False

        self.depth_conv = keras.Sequential([
            layers.DepthwiseConv2D(
                self.kernel_size, self.stride, padding='same'),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.relu6)])
        self.point_linear = keras.Sequential([
            layers.Conv2D(self.out_channels, 1, 1, padding='same'),
            layers.BatchNormalization()])

    def call(self, inputs, training=False):
        if self.use_inverted_block:
            inverted_bottleneck_out = self.inverted_bottleneck(
                inputs, training=training)
        else:
            inverted_bottleneck_out = inputs

        depth_conv_out = self.depth_conv(
            inverted_bottleneck_out, training=training)
        point_out = self.point_linear(
            depth_conv_out, training=training)
        return point_out


class BasicConv2D(layers.Layer):
    def __init__(self, out_channels: int, kernel_size, strides=1, padding='same',
                 activation: bool = True, weight_decay=1e-5, **kwargs):
        super(BasicConv2D, self).__init__()

        self.conv = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                                  use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay), **kwargs)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu') if activation else None
    
    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor, training=training)
        x = self.bn(x, training=training)
        if self.act:
            x = self.act(x, training=training)
        return x


class StemBlock(layers.Layer):
    def __init__(self, init_feats: int):
        super(StemBlock, self).__init__()

        stem_feats = int(init_feats / 2)

        self.stem_basic = BasicConv2D(init_feats, kernel_size=3, strides=2, activation=True)
        self.branch1 = BasicConv2D(stem_feats, kernel_size=1, activation=True)
        self.branch2 = BasicConv2D(init_feats, kernel_size=3, strides=2, activation=True)
        self.pool = layers.MaxPool2D(2)
        self.concat = layers.Concatenate()
        self.last_conv = BasicConv2D(init_feats, kernel_size=1, activation=True)
    
    def call(self, input_tensor, training=False):
        x = self.stem_basic(input_tensor, training=training)
        branch = self.branch2(self.branch1(x), training=training)
        maxpool = self.pool(x, training=training)
        concat = self.concat([branch, maxpool], training=training)
        last_out = self.last_conv(concat, training=training)
        return last_out

class DenseLayer(layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, bottleneck_width: float, expand_ratio: int):
        super(DenseLayer, self).__init__()

        out_channels_2 = out_channels // 2
        inter_channel = int(out_channels_2 * bottleneck_width)

        self.branch1 = keras.Sequential([
            BasicConv2D(inter_channel, kernel_size=1),
            BasicConv2D(out_channels_2, kernel_size=3)])

        self.branch2 = MBInvertedConv(in_channels, out_channels_2, expand_ratio=expand_ratio)
        
        self.concat = layers.Concatenate()
    
    def call(self, input_tensor, training=False):
        branch1 = self.branch1(input_tensor, training=training)
        branch2 = self.branch2(input_tensor, training=training)
        out = self.concat([branch1, branch2], training=training)
        return out

class DenseLayerV2(layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, bottleneck_width: float, expand_ratio: int):
        super(DenseLayer, self).__init__()

        out_channels_2 = out_channels // 2
        inter_channel = int(out_channels_2 * bottleneck_width)

        self.branch1 = keras.Sequential([
            BasicConv2D(inter_channel, kernel_size=1),
            BasicConv2D(out_channels_2, kernel_size=3)])

        self.branch2 = MBInvertedConv(out_channels_2, out_channels, expand_ratio=expand_ratio)
    
    def call(self, input_tensor, training=False):
        branch1 = self.branch1(input_tensor, training=training)
        branch2 = self.branch2(branch1, training=training)
        return branch2


class TransitionLayer(layers.Layer):
    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionLayer, self).__init__()

        self.layer = MBInvertedConv(in_channels, out_channels, 1, 2, expand_ratio=3)
    
    def call(self, input_tensor, training=False):
        layer_out = self.layer(input_tensor, training=training)
        return layer_out

class DenseBlock(layers.Layer):
    def __init__(self, num_layers: int, growth_rate: int, bottleneck_width: int,
                 add_shortcuts: Sequence[bool]):
        super(DenseBlock, self).__init__()

        self.num_layers = num_layers
        self.add_shortcuts = add_shortcuts
        self.layer_name = 'dense_layer_{i}'
        for i in range(num_layers):
            self.__setattr__(self.layer_name.format(i=i), DenseLayer(growth_rate, bottleneck_width))
    
    def call(self, input_tensor, training=False):
        prev_out = input_tensor
        for i in range(self.num_layers):
            out = self.__getattribute__(self.layer_name.format(i=i))(prev_out, training=training)
            if self.add_shortcuts[i]:
                prev_out = tf.add(prev_out, out)
            else:
                prev_out = out
        return prev_out


@BACKBONES.register_module()
class PeleeNet(layers.Layer):
    def __init__(self, stem_feats: int = 32, dense_layers: Sequence[int] = [3, 4, 8],
                 bottleneck_width: Sequence[float] = [1, 2, 4, 4],
                 out_layers: Sequence[int] = [128, 256, 512, 704],
                 growth_rate: Union[int, Sequence[int]] = 32):
        super(PeleeNet, self).__init__(name='PeleeNet')

        self.backbone_name = "[PeleeNet]"

        # Check the lengths
        assert len(dense_layers) == len(bottleneck_width) == len(out_layers), \
            f'Expects same length for dense_layers({dense_layers}), bottleneck_width({bottleneck_width}), ' \
            f'out_layers({out_layers})'
        
        if isinstance(growth_rate, Sequence):
            assert len(growth_rate) == len(dense_layers), f"Expects length of growth_rate same as other inputs"
        else:
            growth_rate = [growth_rate] * len(dense_layers)

        LOG.info(f"{TAG_NAME} {self.backbone_name} Stem Features: {stem_feats}")
        LOG.info(f"{TAG_NAME} {self.backbone_name} Dense Layers: {dense_layers}")
        LOG.info(f"{TAG_NAME} {self.backbone_name} Bottlenect width: {bottleneck_width}")
        LOG.info(f"{TAG_NAME} {self.backbone_name} Output Layers: {out_layers}")
        LOG.info(f"{TAG_NAME} {self.backbone_name} Growth rate: {growth_rate}")

        self.num_layers = len(dense_layers)

        # Stem Block
        self.stem_block = StemBlock(stem_feats)

        self.layer_name = 'layer_{i}'
        for i in range(self.num_layers):
            layer = keras.Sequential([
                DenseBlock(dense_layers[i], growth_rate[i], bottleneck_width[i]),
                TransitionLayer(out_layers[i], use_pooling=(i < 3))])
            self.__setattr__(self.layer_name.format(i=i), layer)
    
    def call(self, input_tensor, training=False):
        x = self.stem_block(input_tensor, training=training)
        outputs = []
        for i in range(self.num_layers):
            x = self.__getattribute__(self.layer_name.format(i=i))(x, training=training)
            outputs.append(x)
        return outputs


if __name__ == "__main__":
    peleenet_model = PeleeNet()
    data = tf.random.uniform([1, 512, 768, 3])
    model_out = peleenet_model(data)
    for out in model_out:
        print(out.shape)
