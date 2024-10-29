from typing import Dict, Sequence
import logging

import tensorflow as tf

from tf_pipeline.models.losses.common import DetLoss, TAG_NAME
from tf_pipeline.utils.registry.models import LOSSES

LOG = logging.getLogger()


@LOSSES.register_module()
class L1Loss(DetLoss):
    def __init__(self, loss_weight: float, type: str):
        super(L1Loss, self).__init__(loss_weight)

        available_types = ['size', 'offset']
        self._name = "[L1Loss]"

        if type not in available_types:
            LOG.error(f'{TAG_NAME} {self._name}: Available types are {available_types}')
            exit(-1)

        LOG.info(f'{TAG_NAME} {self._name}: loss_weight: {loss_weight}, type: {type}')

        self.type = type
    
    @tf.function
    def calculate_loss(self, model_pred, gt_data, indices, mask):
        batch_dim = tf.shape(model_pred)[0]
        channel_dim = tf.shape(model_pred)[-1]
        model_pred = tf.reshape(model_pred, (batch_dim, -1, channel_dim))
        indices = tf.cast(indices, tf.int32)
        model_pred = tf.gather(model_pred, indices, batch_dims=1)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
        total_loss = tf.reduce_sum(tf.abs(gt_data * mask - model_pred * mask))
        loss = total_loss / (tf.reduce_sum(mask) + 1e-5)
        return loss

    @tf.function
    def __call__(self, model_preds: Sequence[tf.Tensor], gt_data: Dict[str, tf.Tensor]) -> tf.Tensor:
        if self.type == 'size':
            loss = self.calculate_loss(model_preds[1], gt_data['size'], gt_data['indices'], gt_data['reg_mask'])
        elif self.type == 'offset':
            loss = self.calculate_loss(model_preds[2], gt_data['offset'], gt_data['indices'], gt_data['reg_mask'])
        else:
            raise NotImplementedError(f'Not implemented for type {self.type}')
        return loss * self.loss_weight
