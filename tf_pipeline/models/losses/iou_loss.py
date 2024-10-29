from typing import Dict, Sequence
import logging

import tensorflow as tf

from tf_pipeline.models.losses.common import DetLoss, TAG_NAME
from tf_pipeline.utils.registry.models import LOSSES

LOG = logging.getLogger()

@tf.function
def get_pos_data(pred_data: tf.Tensor, gt_data: tf.Tensor):
    return

@LOSSES.register_module()
class IoULoss(DetLoss):
    def __init__(self, loss_weight: float):
        super(IoULoss, self).__init__(loss_weight)

        self._name = "[IoULoss]"
        LOG.info(f"{TAG_NAME} {self._name} loss_weight: {loss_weight}")
    
    @tf.function
    def __call__(self, model_preds: Sequence[tf.Tensor], gt_data: Dict[str, tf.Tensor]) -> tf.Tensor:

        return
