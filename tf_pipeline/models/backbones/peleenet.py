import logging
from typing import Union, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import BACKBONES

LOG = logging.getLogger()
TAG_NAME = "[Backbone]"

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
    def __init__(self, growth_rate: int, bottleneck_width: int):
        super(DenseLayer, self).__init__()

        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        self.branch1 = keras.Sequential([
            BasicConv2D(inter_channel, kernel_size=1),
            BasicConv2D(growth_rate, kernel_size=3)])

        self.branch2 = keras.Sequential([
            BasicConv2D(inter_channel, kernel_size=1),
            BasicConv2D(growth_rate, kernel_size=3),
            BasicConv2D(growth_rate, kernel_size=3)
        ])
        
        self.concat = layers.Concatenate()
    
    def call(self, input_tensor, training=False):
        branch1 = self.branch1(input_tensor, training=training)
        branch2 = self.branch2(input_tensor, training=training)
        out = self.concat([input_tensor, branch1, branch2], training=training)
        return out


class TransitionLayer(layers.Layer):
    def __init__(self, out_channels: int, use_pooling: bool = True):
        super(TransitionLayer, self).__init__()

        self.layer = BasicConv2D(out_channels, kernel_size=1)
        self.pooling = layers.AveragePooling2D(2) if use_pooling else None
    
    def call(self, input_tensor, training=False):
        layer_out = self.layer(input_tensor, training=training)
        if self.pooling:
            return self.pooling(layer_out, training=training)
        return layer_out

class DenseBlock(layers.Layer):
    def __init__(self, num_layers: int, growth_rate: int, bottleneck_width: int):
        super(DenseBlock, self).__init__()

        self.num_layers = num_layers
        self.layer_name = 'dense_layer_{i}'
        for i in range(num_layers):
            self.__setattr__(self.layer_name.format(i=i), DenseLayer(growth_rate, bottleneck_width))
    
    def call(self, input_tensor, training=False):
        x = input_tensor
        for i in range(self.num_layers):
            x = self.__getattribute__(self.layer_name.format(i=i))(x, training=training)
        return x


@BACKBONES.register_module()
class PeleeNet(layers.Layer):
    def __init__(self, stem_feats: int = 32, dense_layers: Sequence[int] = [3, 4, 8, 6],
                 bottleneck_width: Sequence[int] = [1, 2, 4, 4],
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
