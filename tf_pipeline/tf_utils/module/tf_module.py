from typing import Dict, List, Sequence
import logging

import tensorflow as tf
import tensorflow.keras as keras

from tf_pipeline.utils.registry.tf_utils import TF_MODULE
from tf_pipeline.utils.registry.models import LOSSES
from tf_pipeline.models.losses.common import DetLoss
from tf_pipeline.utils.configs.config import ModelConfig

LOG = logging.getLogger()
TAG_NAME = "[TFModule]"


class CustomLossContainer(object):
    def __init__(self, losses_config: Dict):
        self.losses: List[DetLoss] = []
        self.means: List[keras.metrics.Mean] = []
        self.loss_sum = keras.metrics.Mean('loss')
        for key, value in losses_config.items():
            loss_fn_name = value.pop("name")
            self.losses.append(LOSSES.get(loss_fn_name)(**value))
            self.means.append(keras.metrics.Mean(key))

    def update_state(self, model_preds: Sequence[tf.Tensor], gt_data: Dict[str, tf.Tensor], regularization_losses=None):
        result = []

        for loss_fn, mean in zip(self.losses, self.means):
            loss = loss_fn(model_preds, gt_data)

            result.append(loss)
            mean.update_state(loss)

        total_loss = tf.add_n(result)
        if regularization_losses:
            reg_loss = tf.add_n(regularization_losses)
            total_loss += reg_loss
            self.loss_sum.update_state(total_loss)
            return reg_loss, total_loss
        else:
            self.loss_sum.update_state(total_loss)
            return 0.0, total_loss


@TF_MODULE.register_module()
class TFModuleV1(keras.Model):
    def __init__(self, *args, **kwargs):
        self.config: ModelConfig = kwargs.pop('config', None)
        self.is_v2_dataset: bool = kwargs.pop('is_v2_dataset', False)
        super(TFModuleV1, self).__init__(*args, **kwargs)

    def compile(self, optimizer='adam', metrics=None,
                run_eagerly=None, steps_per_execution=None, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics,
                        run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, **kwargs)

        # Create Losses
        if len(self.config.losses_config()) == 0:
            LOG.error("No losses found in the config")
            exit(-1)
        self.custom_losses = CustomLossContainer(self.config.losses_config())

    def train_step(self, data):

        with tf.GradientTape() as tape:

            if self.is_v2_dataset:
                model_preds = self(data[0]['input'], training=True)
                reg_loss, loss = self.custom_losses.update_state(
                    model_preds, data[0], regularization_losses=self.losses)
            else:
                model_preds = self(data['input'], training=True)
                reg_loss, loss = self.custom_losses.update_state(
                    model_preds, data, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [(tf.clip_by_norm(grad, 1.0)) for grad in gradients]
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        losses = {m.name: m.result() for m in self.metrics}
        losses["regularization"] = reg_loss
        return losses

    def test_step(self, data):
        if self.is_v2_dataset:
            model_preds = self(data[0]['input'], training=False)
            self.custom_losses.update_state(
                model_preds, data[0], regularization_losses=self.losses)
        else:
            model_preds = self(data['input'], training=False)
            self.custom_losses.update_state(
                model_preds, data, regularization_losses=self.losses)
        return_dict = {m.name: m.result() for m in self.metrics}
        return return_dict

    @property
    def metrics(self):
        metrics = super().metrics
        if self.custom_losses:
            metrics.append(self.custom_losses.loss_sum)
            metrics.extend(self.custom_losses.means)
        return metrics
