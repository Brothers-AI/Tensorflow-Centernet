import logging
from typing import Union, Sequence

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tf_pipeline.utils.registry.models import NECKS

LOG = logging.getLogger()
TAG_NAME = "[Neck]"


@NECKS.register_module()
class DeconvLayer(layers.Layer):
    def __init__(self, deconv_channels: Sequence[int], deconv_kernels: Sequence[int],
                 idx_to_take: int = -1):
        super(DeconvLayer, self).__init__(name='DeconvLayer')

        self.neck_name = "[DeconvLayer]"

        assert len(deconv_channels) == len(
            deconv_kernels), f"Expects same shape for deconv_channels ({deconv_channels}) and deconv kernels {deconv_kernels}"

        LOG.info(f"{TAG_NAME} {self.neck_name} deconv_channels: {deconv_channels}")
        LOG.info(f"{TAG_NAME} {self.neck_name} deconv_kernels: {deconv_kernels}")
        LOG.info(f"{TAG_NAME} {self.neck_name} idx_to_take: {idx_to_take}")

        self.deconv_module_name = "deconv_{idx}"
        self.idx_to_take = idx_to_take
        self.num_layers = len(deconv_channels)

        for idx, (deconv_out_ch, deconv_k) in enumerate(zip(deconv_channels, deconv_kernels)):
            deconv_module = keras.Sequential([
                layers.Conv2DTranspose(deconv_out_ch, kernel_size=deconv_k,
                                       strides=2, padding='same', use_bias=False,
                                       kernel_initializer='he_uniform',
                                       kernel_regularizer=keras.regularizers.l2(1.25e-5)),
                layers.BatchNormalization(),
                layers.Activation('relu')])
            self.__setattr__(self.deconv_module_name.format(
                idx=idx), deconv_module)

    def call(self, input_tensors, training=False):
        x = input_tensors[self.idx_to_take]
        for idx in range(self.num_layers):
            x = self.__getattribute__(
                self.deconv_module_name.format(idx=idx))(x, training=training)
        return x
