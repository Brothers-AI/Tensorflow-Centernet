import logging
from typing import Tuple, Union, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

TAG_NAME = ['TF-Transforms']
LOG = logging.getLogger()

class Normalize(layers.Layer):
    def __init__(self, mean: Union[List, np.ndarray], std: Union[List, np.ndarray]):
        super(Normalize, self).__init__(trainable=False, name='Normalize')

        self.mean = tf.convert_to_tensor(np.array(mean, dtype=np.float32), dtype=tf.float32)
        self.std = tf.convert_to_tensor(np.array(std, dtype=np.float32), dtype=tf.float32)
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        image = tf.divide(tf.subtract(image, self.mean), self.std)
        return image, bboxes, labels