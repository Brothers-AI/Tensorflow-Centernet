import logging
from typing import Union, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import NECKS

LOG = logging.getLogger()
TAG_NAME = "[Neck]"

class BiFPNModule(layers.Layer):
    def __init__(self, out_channels: int,
                 levels: int,
                 kernel_size: int,
                 depth_multiplier: int):
        super(BiFPNModule, self).__init__(name='bifpn_module')

        self.w1 = self.add_weight(name='sum_weights_1',
                                  shape=(2, levels),
                                  initializer='ones',
                                  trainable=True)
        self.w2 = self.add_weight(name='')

@NECKS.register_module()
class BiFPN(layers.Layer):
    def __init__(self, num_inputs: int,
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 depth: int = 1,
                 kernel_size: int = 3,
                 depth_multiplier: int = 1,
                 pooling_strategy: str = 'avg'):
        super(BiFPN, self).__init__(name='bifpn')

        self.depth = depth
        self.num_inputs = num_inputs
        self.start_level = start_level

        assert start_level >= 0, f"start level ({start_level}) must be greater than or equal to 0"

        if end_level == -1:
            self.backbone_end_level = self.num_inputs
            assert num_outs == num_inputs - start_level, f"Number of outputs passesd ({num_outs}) and \
                                                           number of inputs calculated ({num_inputs - start_level}) are \
                                                           not same"
        else:
            self.backbone_end_level = end_level
            assert end_level <= self.num_inputs
            assert num_outs == end_level - start_level

        assert num_outs <= num_inputs, f"Expected outputs ({num_outs}) to be less than or \
                                         equal to inputs ({num_inputs})"

        # Lateral Convolutions
        self.lateral_convs = []
        for i in range(start_level, self.backbone_end_level):
            l_conv = keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=1, padding='same'),
                layers.BatchNormalization(),
                layers.Activation(tf.nn.relu6)])
            self.lateral_convs.append(l_conv)
        
        assert num_outs == len(self.lateral_convs), f"Number of lateral convs ({len(self.lateral_convs)}) \
                                                      is not same as number of outputs ({num_outs})"
        


