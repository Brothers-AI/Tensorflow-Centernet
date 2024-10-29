import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

TAG_NAME = ['TF-Transforms']
LOG = logging.getLogger()


class RandomColorSpace(layers.Layer):
    def __init__(self, prob: float = 0.5):
        super(RandomColorSpace, self).__init__(trainable=False, name='RandomColorSpace')

        LOG.info(f"{TAG_NAME} [RandomColorSpace]: prob -> {prob}")
        self.prob = prob
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        if np.random.random() < self.prob:
            image = tf.reverse(image, axis=[-1])
        return image, bboxes, labels

class RandomBlur(layers.Layer):
    def __init__(self, prob: float = 0.5):
        super(RandomBlur, self).__init__(trainable=False, name='RandomBlur')

        LOG.info(f"{TAG_NAME} [RandomBlur]: prob -> {prob}")
        self.prob = prob

        # Blur
        blur_kernel = tf.constant(1.0 / (5 * 5), shape=(5, 5))
        self.blur_kernel = tf.reshape(blur_kernel, [5, 5, 1, 1])

        # Gaussian Blur
        kernel_size = 5
        sigma = 5
        kernel = tf.exp(-tf.square(tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)) / (2.0 * sigma ** 2))
        self.gauss_kernel = kernel / tf.reduce_sum(kernel)

        self.funs = [self.avg_blur, self.gaussian_blur]

    def avg_blur(self, image):
        return tf.nn.conv2d(image, self.blur_kernel, strides=[1, 1, 1, 1], padding='SAME')

    def gaussian_blur(self, image):
        return tf.nn.conv2d(image, self.gauss_kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        if np.random.random() < self.prob:
            idx = np.random.randint(0, len(self.funs))
            image = self.funs[idx](image)
        return image, bboxes, labels

class RandomBrightnessContrast(layers.Layer):
    def __init__(self, prob: float = 0.5):
        super(RandomBrightnessContrast, self).__init__(trainable = False, name = 'RandomBrightnessContrast')

        LOG.info(f"{TAG_NAME} [RandomBrightnessContrast]: prob -> {prob}")
        self.prob = prob

        self.funcs = [self.brightness, self.contrast]
    
    def brightness(self, image):
        return tf.image.adjust_brightness(image, delta=0.2)
    
    def contrast(self, image):
        return tf.image.adjust_contrast(image, contrast_factor=0.3)
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        if np.random.random() < self.prob:
            idx = np.random.randint(0, len(self.funcs))
            image = self.funcs[idx](image)
        return image, bboxes, labels

class RandomGamma(layers.Layer):
    def __init__(self, prob: float = 0.5):
        super(RandomGamma, self).__init__(trainable = False, name = 'RandomGamma')

        LOG.info(f"{TAG_NAME} [RandomGamma]: prob -> {prob}")
        self.prob = prob
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        if np.random.random() < self.prob:
            image = tf.image.adjust_gamma(image, 0.5)
        return image, bboxes, labels


class RandomQuality(layers.Layer):
    def __init__(self, prob: float = 0.5):
        super(RandomQuality, self).__init__(trainable = False, name = 'RandomQuality')

        LOG.info(f"{TAG_NAME} [RandomQuality]: prob -> {prob}")
        self.prob = prob
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        if np.random.random() < self.prob:
            image = tf.image.adjust_jpeg_quality(image, 80)
        return image, bboxes, labels


class RandomHSV(layers.Layer):
    def __init__(self, prob: float = 0.5):
        super(RandomHSV, self).__init__(trainable = False, name = 'RandomHSV')

        LOG.info(f"{TAG_NAME} [RandomHSV]: prob -> {prob}")
        self.prob = prob
    
    def call(self, image: tf.Tensor, bboxes: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        bboxes = tf.reshape(bboxes, (-1, 4))

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, (-1, 1))

        if np.random.random() < self.prob:
            image = tf.image.adjust_hue(image, 0.2)
        return image, bboxes, labels