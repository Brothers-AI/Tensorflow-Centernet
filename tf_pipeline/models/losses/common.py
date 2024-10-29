from typing import Dict, Sequence

import tensorflow as tf

TAG_NAME = "[DetLoss]"

class DetLoss(object):
    def __init__(self, loss_weight: float):
        self.loss_weight = tf.constant(loss_weight, dtype=tf.float32)
    
    @tf.function
    def __call__(self, model_preds: Sequence[tf.Tensor], gt_data: Dict[str, tf.Tensor]) -> tf.Tensor:
        raise NotImplementedError('To be implemented by child class')