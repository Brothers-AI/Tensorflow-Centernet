from typing import Union, Dict, Any, Tuple, Sequence
import os
import json


class DotDict(dict):

    __getattr__ = dict.__getitem__

    def __init__(self, d: dict):
        self.update(**dict((k, self.parse(v))
                           for k, v in d.items()))

    @classmethod
    def parse(cls, v):
        if isinstance(v, dict):
            return cls(v)
        elif isinstance(v, list):
            return [cls.parse(i) for i in v]
        else:
            return v


class ModelConfig(object):
    def __init__(self, model_config: dict):
        self.model_config: DotDict = DotDict(model_config)

        self._is_backbone_available = False
        self._is_neck_available = False
        self._is_head_available = False
        self._is_auxilary_neck_available = False

        # Check for backbone
        if "backbone" in model_config.keys():
            self._is_backbone_available = True

        # Check for neck
        if "neck" in model_config.keys():
            self._is_neck_available = True

        # Check for Auxilary neck
        if "auxilary_neck" in model_config.keys():
            self._is_auxilary_neck_available = True

        # Check for Head
        if "head" in model_config.keys():
            self._is_head_available = True
        
        # Check for losses
        if "losses" not in model_config.keys():
            raise KeyError(f"Expected \"losses\" key in model config, But not found.")
        losses_config = self.model_config.losses
        if not isinstance(losses_config, dict):
            raise TypeError(f"Expected losses type to be of dict, but found of type {type(losses_config)}")
        self._loss_config = losses_config

        # Check for post proc
        if "post_proc" not in model_config.keys():
            raise KeyError(f"Expected \"post_proc\" key in model config, But not found.")
        postproc_config = self.model_config.post_proc
        if not isinstance(postproc_config, dict):
            raise TypeError(f"Expected post_proc type to be of dict, but found of type {type(postproc_config)}")
        self._postproc_config = postproc_config

    @property
    def is_backbone_available(self) -> bool:
        return self._is_backbone_available

    @property
    def is_neck_available(self) -> bool:
        return self._is_neck_available

    @property
    def is_head_available(self) -> bool:
        return self._is_head_available

    @property
    def is_auxilary_neck_available(self) -> bool:
        return self._is_auxilary_neck_available

    def postproc_config(self) -> Tuple[str, Union[DotDict, dict]]:
        postproc_name = self._postproc_config.pop('name')
        return postproc_name, self._postproc_config

    def losses_config(self) -> Union[DotDict, Dict]:
        return self._loss_config

    def backbone_config(self) -> Tuple[Union[None, str], Union[None, DotDict, Dict]]:
        if self.is_backbone_available:
            backbone_config = self.model_config.backbone
            if not (isinstance(backbone_config, DotDict) or isinstance(backbone_config, dict)):
                raise TypeError(
                    f"Expected backbone type to be dict / DotDict, but found type {type(backbone_config)}")
            backbone_name = backbone_config.pop('name')
            return backbone_name, backbone_config
        else:
            return None, None

    def neck_config(self) -> Tuple[Union[None, str], Union[None, DotDict, Dict]]:
        if self.is_neck_available:
            neck_config = self.model_config.neck
            if not (isinstance(neck_config, DotDict) or isinstance(neck_config, dict)):
                raise TypeError(
                    f"Expected neck type to be dict / DotDict, but found type {type(neck_config)}")
            neck_name = neck_config.pop('name')
            return neck_name, neck_config
        else:
            return None, None

    def auxilary_neck_config(self) -> Tuple[Union[None, str], Union[None, DotDict, Dict]]:
        if self.is_auxilary_neck_available:
            auxilary_neck_config = self.model_config.auxilary_neck
            if not (isinstance(auxilary_neck_config, DotDict) or isinstance(auxilary_neck_config, dict)):
                raise TypeError(
                    f"Expected auxilary neck type to be dict / DotDict, but found type {type(auxilary_neck_config)}")
            auxilary_neck_name = auxilary_neck_config.pop('name')
            return auxilary_neck_name, auxilary_neck_config
        else:
            return None, None

    def head_config(self) -> Tuple[Union[None, str], Union[None, DotDict, Dict]]:
        if self.is_head_available:
            head_config = self.model_config.head
            if not (isinstance(head_config, DotDict) or isinstance(head_config, dict)):
                raise TypeError(
                    f"Expected head type to be dict / DotDict, but found type {type(head_config)}")
            head_name = head_config.pop('name')
            return head_name, head_config
        else:
            return None

class TrainParams(object):
    def __init__(self, trainer_config: dict):
        self.trainer_config = DotDict(trainer_config)

        # Check for Learning rate
        if "lr" not in trainer_config.keys():
            raise KeyError("Expected \"lr\" in train config, but not found")
        
        # Check for Learning rate Scheduler
        if "lr_scheduler" not in trainer_config.keys():
            raise KeyError("Expected \"lr_scheduler\" in train config, but not found")
        
        # Check for batch size
        if "batch_size" not in trainer_config.keys():
            raise KeyError("Expected \"batch_size\" in train config, but not found")
        
        # Check for Num workers
        if "num_workers" not in trainer_config.keys():
            raise KeyError("Expected \"num_workers\" in train config, but not found")
        
        # Check for Num Epochs
        if "num_epochs" not in trainer_config.keys():
            raise KeyError("Expected \"num_epochs\" in train config, but not found")
        
        # Check for Eval Freq
        if "eval_freq" not in trainer_config.keys():
            raise KeyError("Expected \"eval_freq\" in train config, but not found")
        
        # Check for dataset
        if "dataset" not in trainer_config.keys():
            raise KeyError("Expected \"dataset\" in train config, but not found")
        self._dataset = self.trainer_config.dataset
        if not isinstance(self._dataset, dict):
            raise TypeError(
                f"Expected dataset type to be dict / DotDict, but found type {type(self._dataset)}")

        # Check for transforms
        if "transforms" not in trainer_config.keys():
            raise KeyError("Expected \"transforms\" in train config, but not found")
        self._transforms = self.trainer_config.transforms
        if not isinstance(self._transforms, dict):
            raise TypeError(
                f"Expected transforms type to be dict / DotDict, but found type {type(self._transforms)}")
    
    @property
    def eval_freq(self):
        return self.trainer_config.eval_freq

    @property
    def lr(self) -> float:
        return float(self.trainer_config.lr)
    
    @property
    def lr_scheduler(self) -> str:
        return self.trainer_config.lr_scheduler
    
    @property
    def batch_size(self) -> int:
        return int(self.trainer_config.batch_size)
    
    @property
    def num_workers(self) -> int:
        return int(self.trainer_config.num_workers)
    
    @property
    def num_epochs(self) -> int:
        return int(self.trainer_config.num_epochs)
    
    def dataset_config(self) -> Tuple[str, Dict]:
        dataset_name = self._dataset.pop("name")
        return dataset_name, self._dataset
    
    @property
    def transforms(self) -> Union[dict, DotDict]:
        return self._transforms

class ValParams(object):
    def __init__(self, val_config: dict):
        self.val_config = DotDict(val_config)

        # Check for Num workers
        if "num_workers" not in val_config.keys():
            raise KeyError("Expected \"num_workers\" in val config, but not found")
        
        # Check for dataset
        if "dataset" not in val_config.keys():
            raise KeyError("Expected \"dataset\" in val config, but not found")
        self._dataset = self.val_config.dataset
        if not isinstance(self._dataset, dict):
            raise TypeError(
                f"Expected dataset type to be dict / DotDict, but found type {type(self._dataset)}")
        
        # Check for transforms
        if "transforms" not in val_config.keys():
            raise KeyError("Expected \"transforms\" in val config, but not found")
        self._transforms = self.val_config.transforms
        if not isinstance(self._transforms, dict):
            raise TypeError(
                f"Expected transforms type to be dict / DotDict, but found type {type(self._transforms)}")
    
    @property
    def num_workers(self) -> int:
        return int(self.val_config.num_workers)
    
    @property
    def batch_size(self) -> int:
        return int(1)
    
    def dataset_config(self) -> Tuple[str, Dict]:
        dataset_name = self._dataset.pop("name")
        return dataset_name, self._dataset
    
    @property
    def transforms(self) -> Union[dict, DotDict]:
        return self._transforms

class Config(object):
    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError(
                f"Expects dictonary as input to Config, but found of type ({type(config)})")

        self.config = DotDict(config)

        # Check for model
        if "model" not in config.keys():
            raise KeyError("Expected \"model\" in config, but not found")
        model_params = self.config.model
        if not isinstance(model_params, dict):
            raise TypeError(
                f"Expected model type to be dict / DotDict, but found type {type(model_params)}")
        self._model_config = ModelConfig(model_params)

        # Check for model resolution
        if "model_resolution" not in config.keys():
            raise KeyError(
                "Expected \"model_resolution\" in config, but not found")
        self._model_resolution = self.config.model_resolution
        if len(self._model_resolution) != 2:
            raise ValueError(
                f"Expected only 2 values [H, W] in \"model_resolution\", but found {len(self._model_resolution)}")

        # Check for save dir
        if "save_dir" not in config.keys():
            raise KeyError("Expected \"save_dir\" in config, but not found")
        self._save_dir = self.config.save_dir

        # Check for exp_name
        if "exp_name" not in config.keys():
            raise KeyError("Expected \"exp_name\" in config, but not found")
        self._exp_name = self.config.exp_name

        # Check for ckpt-path
        if "ckpt_path" not in config.keys():
            raise KeyError("Expected \"ckpt_path\" in config, but not found")
        self._ckpt_path = self.config.ckpt_path

        # Check for device
        if "device" not in config.keys():
            raise KeyError("Expected \"device\" in config, but not found")
        self._device = self.config.device

        # Check for train params
        if "train_params" not in config.keys():
            raise KeyError("Expected \"train_params\" in config, but not found")
        self._train_params = TrainParams(self.config.train_params)

        # Check for val params
        if "val_params" not in config.keys():
            raise KeyError("Expected \"val_params\" in config, but not found")
        self._val_params = ValParams(self.config.val_params)

        # Check for tf_module
        if "tf_module" not in config.keys():
            raise KeyError("Expected \"tf_module\" in config, but not found")
        self._tf_module = self.config.tf_module

        # Check for resume
        if "resume" not in config.keys():
            raise KeyError("Expected \"resume\" in config, but not found")
        self._resume = self.config.resume

        # Check for pretrained_path
        if "pretrained_path" not in config.keys():
            raise KeyError("Expected \"pretrained_path\" in config, but not found")
        self._pretrained_path = self.config.pretrained_path

        # Check for pretrained_saved_path
        if "pretrained_saved_path" not in config.keys():
            raise KeyError("Expected \"pretrained_saved_path\" in config (Saved model path), but not found")
        self._pretrained_saved_path = self.config.pretrained_saved_path

        # Check for initial_epoch
        if "initial_epoch" not in config.keys():
            raise KeyError("Expected \"initial_epoch\" in config, but not found")
        self._initial_epoch = self.config.initial_epoch

    @property
    def initial_epoch(self):
        return self._initial_epoch

    @property
    def pretrained_path(self):
        return self._pretrained_path
    
    @property
    def pretrained_saved_path(self):
        return self._pretrained_saved_path

    @property
    def resume(self):
        return self._resume

    @property
    def tf_module(self):
        return self._tf_module

    @property
    def model_resolution(self) -> Sequence[int]:
        return self._model_resolution

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def exp_name(self):
        return self._exp_name

    @property
    def ckpt_path(self):
        return self._ckpt_path

    @property
    def device(self):
        return self._device
    
    @property
    def train_params(self) -> TrainParams:
        return self._train_params

    @property
    def val_params(self) -> ValParams:
        return self._val_params

    def __str__(self) -> str:
        return json.dumps(self.config, indent=4)

    @staticmethod
    def parse_file(filepath):
        assert os.path.exists(
            filepath), f'Config file {filepath} doesn\'t found. Please check'
        with open(filepath, "r") as fp:
            config_data = json.load(fp)
        return Config(config_data)
