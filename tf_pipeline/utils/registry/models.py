from .registry import Registry

BACKBONES = Registry("backbones", location="tf_pipeline.models.backbones")
NECKS = Registry("necks", location="tf_pipeline.models.necks")
HEADS = Registry("heads", location="tf_pipeline.models.heads")
LOSSES = Registry("losses", location="tf_pipeline.models.losses")