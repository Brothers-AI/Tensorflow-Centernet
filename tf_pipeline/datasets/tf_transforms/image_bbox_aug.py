import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

TAG_NAME = ['TF-Transforms']
LOG = logging.getLogger()

class RandomHFlip(layers.Layer):
    def __init__(self, prob: float = 0.5):
        super(RandomHFlip, self).__init__(trainable = False, name = 'RandomHFlip')

        self.prob = prob
        LOG.info(f"{TAG_NAME} [RandomHFlip]: prob -> {prob}")
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        if np.random.random() < self.prob:
            image = tf.reverse(image, axis=-1)

            W = tf.shape(image)[2]
            H = tf.shape(image)[1]
            constant_data = tf.constant([W, W], dtype=tf.float32)
            x_data = tf.stack([bboxes[:, 0], bboxes[:, 2]], axis=1)
            x_data = tf.subtract(constant_data, x_data)
            x_data = tf.clip_by_value(x_data, clip_value_min=0, clip_value_max=W)

            y_data = tf.stack([bboxes[:, 1], bboxes[:, 3]], axis=1)
            y_data = tf.clip_by_value(y_data, clip_value_min=0, clip_value_max=H)
            bboxes = tf.stack([x_data[:, 0], y_data[:, 0], x_data[:, 1], y_data[:, 1]], axis=1)
        return image, bboxes, labels