from typing import Dict, Sequence
import logging

import tensorflow as tf

from tf_pipeline.models.losses.common import DetLoss, TAG_NAME
from tf_pipeline.utils.registry.models import LOSSES

LOG = logging.getLogger()

@LOSSES.register_module()
class FocalLoss(DetLoss):
    def __init__(self, loss_weight: float):
        super(FocalLoss, self).__init__(loss_weight)

        self._name = "[FocalLoss]"
        LOG.info(f"{TAG_NAME} {self._name} loss_weight: {loss_weight}")
    
    @tf.function
    def __call__(self, model_preds: Sequence[tf.Tensor], gt_data: Dict[str, tf.Tensor]) -> tf.Tensor:
        heatmap_pred: tf.Tensor = model_preds[0]
        heatmap_gt: tf.Tensor = gt_data['heatmap']

        pos_mask = tf.cast(tf.equal(heatmap_gt, 1.0), dtype=tf.float32)
        neg_mask = tf.cast(tf.less(heatmap_gt, 1.0), dtype=tf.float32)
        neg_weights = tf.pow(1.0 - heatmap_gt, 4)

        pos_loss = -tf.math.log(tf.clip_by_value(heatmap_pred, 1e-5, 1.0 - 1e-5)) * tf.math.pow(1.0 - heatmap_pred, 2.0) * pos_mask
        neg_loss = -tf.math.log(tf.clip_by_value(1.0 - heatmap_pred, 1e-5, 1.0 - 1e-5)) \
                    * tf.math.pow(heatmap_pred, 2.0) * neg_weights * neg_mask
        
        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)

        loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
        return loss * self.loss_weight
