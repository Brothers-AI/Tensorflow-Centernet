import logging

import tensorflow as tf
import tensorflow.keras as keras

from tf_pipeline.models.build import build_backbone, build_neck, build_head
from tf_pipeline.utils.configs.config import ModelConfig

LOG = logging.getLogger()
TAG_NAME = "[Model]"

class Identity(keras.layers.Layer):
    def __init__(self, name: str):
        super(Identity, self).__init__(name=name)
    
    def call(self, inputs, training=False):
        return inputs

class DetectionModel(keras.layers.Layer):
    def __init__(self, config: ModelConfig, num_classes: int):
        super(DetectionModel, self).__init__()
        self.config = config

        # Backbone
        if config.is_backbone_available:
            backbone_name, backbone_kwargs = config.backbone_config()
            self.backbone = build_backbone(backbone_name, backbone_kwargs)
        else:
            LOG.warning('Backbone is not available in config. Creating Identity layer')
            self.backbone = Identity('backbone_identity')
        
        # Neck
        if config.is_neck_available:
            neck_name, neck_kwargs = config.neck_config()
            self.neck = build_neck(neck_name, neck_kwargs)
        else:
            LOG.warning('Neck is not available in config. Creating Identity layer')
            self.neck = Identity('neck_identity')
        
        # Auxilary Neck
        if config.is_auxilary_neck_available:
            aux_neck_name, aux_neck_kwargs = config.auxilary_neck_config()
            self.aux_neck = build_neck(aux_neck_name, aux_neck_kwargs)
        else:
            LOG.warning('Aux Neck is not available in config. Creating Identity layer')
            self.aux_neck = Identity('aux_neck_identity')
        
        # Head
        if config.is_head_available:
            head_name, head_kwargs = config.head_config()
            head_kwargs["num_classes"] = num_classes
            self.head = build_head(head_name, head_kwargs)
        else:
            LOG.warning('Head is not available in config. Creating Identity layer')
            self.head = Identity('head_identity')
    
    def call(self, inputs, training=False):
        backbone_out = self.backbone(inputs, training=training)
        neck_out = self.neck(backbone_out, training=training)
        aux_neck_out = self.aux_neck(neck_out, training=training)
        head_out = self.head(aux_neck_out, training=training)
        return head_out
    
    def build(self, input_shape):
        input_layer = keras.Input(shape=input_shape, name='input')
        return keras.Model(inputs=[input_layer], outputs=self.call(input_layer))

