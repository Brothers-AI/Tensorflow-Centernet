import os
import sys
import time
import logging
from typing import Tuple

# For Showing only required Memory in GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# For Less CPU Load = 4 * NUM_GPUS
os.environ['OMP_NUM_THREAD'] = '4'

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import BatchDataset

from tf_pipeline.utils.registry.datasets import DATASETS
from tf_pipeline.datasets import DefaultDataset, DefaultDatasetV2
from tf_pipeline.utils.configs.config import Config, TrainParams, ValParams
from tf_pipeline.models.create import create_tf_model
from tf_pipeline.callbacks import LoggerCallback, TensorboardCallback, ValProgressBar
from tf_pipeline.lr_schedulers.schedulers import DefaultStepScheduler, CosineScheduler, \
                                                 CosineFlatScheduler, LinearScheduler


AVAILABLE_LR_SCHEDULERS = ['DefaultStep', 'Cosine', 'ReduceLROnPlateau']


class TensorflowTrainer(object):
    def __init__(self, config_path: str):

        config = Config.parse_file(config_path)

        # Create directories
        self.directories_dict = self.create_directories(config)
        self.logger = logging.getLogger()

        self.logger.info(f"Config file -> {config_path}")
        self.logger.info(f"Config file contents\n{config}")

        # Copy the config to exp_dir
        exp_save_dir = self.directories_dict['save_dir']
        os.system(f"cp {config_path} {exp_save_dir}")

        # is V2 Dataset
        self.is_v2_dataset = False

        # Create Train dataset
        self.train_params, self.train_dataset, self.train_ds, self.num_train_iters = self.initialize_train_dataset(
            config)

        # Create val dataset
        self.val_params, self.val_dataset, self.val_ds, self.num_val_iters = self.initalize_val_dataset(
            config)

        # Get the epoch and ckpt path
        self.initial_epoch, latest_ckpt_path = self.get_epoch(config, self.directories_dict['checkpoints_dir'])

        if self.initial_epoch < 0:
            raise ValueError(
                "Resume is given with negative value, Please give postive value to resume")

        self.logger.info(f"Initial Epoch: {self.initial_epoch}")

        # Create optimizer and scheduler
        optimizer, scheduler_cb = self.create_optimizer_and_scheduler(
            self.initial_epoch)

        # Create model
        self.model = self.create_model(config, optimizer, latest_ckpt_path)

        # Initalize callbacks
        self.callbacks = self.initalize_callbacks(scheduler_cb)

    def create_directories(self, config: Config):

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        exp_save_dir = os.path.join(config.save_dir, config.exp_name)
        logs_dir = os.path.join(exp_save_dir, "logs")
        log_filename = f"logfile_{time_str}.log"
        tensorboard_logs_dir = os.path.join(exp_save_dir, "tensorboard_logs")
        checkpoints_dir = os.path.join(exp_save_dir, "checkpoints")

        directories_dict = {
            "save_dir": exp_save_dir,
            "logs_dir": logs_dir,
            "tensorboard_dir": tensorboard_logs_dir,
            "checkpoints_dir": checkpoints_dir
        }

        # Create directories
        os.makedirs(exp_save_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(tensorboard_logs_dir, exist_ok=True)

        # Logging config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(logs_dir, log_filename)),
                logging.StreamHandler(sys.stdout)
            ]
        )

        return directories_dict

    def initialize_train_dataset(self, config: Config) -> Tuple[TrainParams, DefaultDataset, BatchDataset]:
        # Get the train params
        train_params = config.train_params
        train_dataset_name, train_dataset_kwargs = train_params.dataset_config()

        # Get the transforms
        train_transforms = train_params.transforms

        if 'V2' in train_dataset_name:
            self.is_v2_dataset = True
            # Create the train dataset
            train_dataset_kwargs['batch_size'] = train_params.batch_size
            train_dataset_kwargs['shuffle'] = True
            train_dataset: DefaultDatasetV2 = DATASETS.get(train_dataset_name)(model_resolution=config.model_resolution,
                                                                                split="train", transforms=train_transforms,
                                                                                init_lr=train_params.lr,
                                                                                **train_dataset_kwargs)
            
            train_ds = train_dataset
            num_iters = len(train_dataset)
        else:
            # Create the train dataset
            train_dataset: DefaultDataset = DATASETS.get(train_dataset_name)(model_resolution=config.model_resolution,
                                                                            split="train", transforms=train_transforms,
                                                                            init_lr=train_params.lr,
                                                                            **train_dataset_kwargs)
            train_ds = tf.data.Dataset.from_generator(train_dataset, output_types=train_dataset.output_types,
                                                    output_shapes=train_dataset.output_shapes)
            options = tf.data.Options()
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            train_ds = train_ds.with_options(options)
            train_ds = train_ds.batch(train_params.batch_size, drop_remainder=True)

            num_iters = int(len(train_dataset) // train_params.batch_size)

        return train_params, train_dataset, train_ds, num_iters

    def initalize_val_dataset(self, config: Config) -> Tuple[ValParams, DefaultDataset, BatchDataset]:
        # Get the val params
        val_params = config.val_params
        val_dataset_name, val_dataset_kwargs = val_params.dataset_config()

        # Get the transforms
        val_transforms = val_params.transforms

        if 'V2' in val_dataset_name:
            self.is_v2_dataset = True
            # Create the val dataset
            val_dataset_kwargs['batch_size'] = 1
            val_dataset_kwargs['shuffle'] = False
            val_dataset: DefaultDatasetV2 = DATASETS.get(val_dataset_name)(model_resolution=config.model_resolution,
                                                                            split="val", transforms=val_transforms,
                                                                            **val_dataset_kwargs)
            
            val_ds = val_dataset
            num_iters = len(val_dataset)
        else:
            # Create the val dataset
            val_dataset: DefaultDataset = DATASETS.get(val_dataset_name)(model_resolution=config.model_resolution,
                                                                        split="val", transforms=val_transforms,
                                                                        **val_dataset_kwargs)
            val_ds = tf.data.Dataset.from_generator(val_dataset, output_types=val_dataset.output_types,
                                                    output_shapes=val_dataset.output_shapes)
            options = tf.data.Options()
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            val_ds = val_ds.with_options(options)
            val_ds = val_ds.batch(val_params.batch_size, drop_remainder=True)

            num_iters = int(len(val_dataset) // val_params.batch_size)

        return val_params, val_dataset, val_ds, num_iters

    def get_epoch(self, config: Config, checkpoints_dir: str):
        initial_epoch = config.initial_epoch
        latest_ckpt_path = None
        if config.resume:
            latest_ckpt_path = tf.train.latest_checkpoint(checkpoints_dir)
            if latest_ckpt_path:
                initial_epoch = int(str(latest_ckpt_path).split("_")[-1])
        return initial_epoch, latest_ckpt_path

    def create_optimizer_and_scheduler(self, initial_epoch: int):

        # Check for available lr schedulers
        if self.train_params.lr_scheduler not in AVAILABLE_LR_SCHEDULERS:
            raise ValueError(f"{self.train_params.lr_scheduler} is not in available lr schedulers: {AVAILABLE_LR_SCHEDULERS}")

        # Scheduler callback
        if self.train_params.lr_scheduler == 'Cosine':
            scheduler = CosineScheduler(self.train_params.num_epochs,
                                        self.train_params.lr,
                                        0.01)
            scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler.scheduler, verbose=1)
            optimizer = tf.keras.optimizers.Adam(scheduler.scheduler(initial_epoch))
        elif self.train_params.lr_scheduler == 'DefaultStep':
            scheduler = DefaultStepScheduler(self.train_params.lr, [90, 120])
            scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler.scheduler, verbose=1)
            optimizer = tf.keras.optimizers.Adam(scheduler.scheduler(initial_epoch))
        elif self.train_params.lr_scheduler == 'CosineFlat':
            scheduler = CosineFlatScheduler(self.train_params.num_epochs,
                                            self.train_params.lr,
                                            0.01)
            scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler.scheduler, verbose=1)
            optimizer = tf.keras.optimizers.Adam(scheduler.scheduler(initial_epoch))
        elif self.train_params.lr_scheduler == 'Linear':
            scheduler = LinearScheduler(self.train_params.num_epochs,
                                        self.train_params.lr,
                                        0.01)
            scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler.scheduler, verbose=1)
            optimizer = tf.keras.optimizers.Adam(scheduler.scheduler(initial_epoch))
        elif self.train_params.lr_scheduler == 'ReduceLROnPlateau':
            scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                factor=0.1,
                                                                patience=5,
                                                                verbose=1,
                                                                min_lr=1e-8)
            optimizer = tf.keras.optimizers.Adam(self.train_params.lr)
        else:
            raise NotImplementedError(f'Not implemented for scheduler {self.train_params.lr_scheduler}')
        
        self.logger.info(f"LR Scheduler -> {self.train_params.lr_scheduler}")

        return optimizer, scheduler_cb

    def create_model(self, config: Config, optimizer: tf.keras.optimizers.Optimizer, latest_ckpt_path: str):

        # For Multi GPU
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            # Create the TF Model
            model = create_tf_model(config.tf_module, config.model_config, [
                *config.model_resolution, 3], num_classes=self.train_dataset.num_classes, is_v2_dataset=self.is_v2_dataset)

            if config.pretrained_path:
                self.logger.info(f"Loading pretrained weights from {config.pretrained_path}")
                model.load_weights(config.pretrained_path).expect_partial()
            
            if config.pretrained_saved_path:
                self.logger.info(f"Loading pretrained weights from saved model {config.pretrained_saved_path}")
                saved_model = tf.keras.models.load_model(config.pretrained_saved_path)
                # Set the weights
                model.set_weights(saved_model.get_weights())
                self.logger.info(f"Loaded the weights from saved model")

            if config.resume:
                if latest_ckpt_path:
                    self.logger.info(
                        f"Resuming from checkpoint {config.ckpt_path}")
                    model.load_weights(latest_ckpt_path).expect_partial()
                else:
                    self.logger.error(
                        f"Checkpoint path not given, Not loading previous checkpoint weights")

            # Compile the model
            model.compile(optimizer=optimizer)
            model.summary()
        return model

    def initalize_callbacks(self, scheduler_cb):

        # Model Checkpoint
        model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                self.directories_dict['checkpoints_dir'], "model_{epoch}"),
            save_freq="epoch",
            verbose=1,
            save_best_only=False,
            save_weights_only=True)

        # Tensorboard callback
        tensorboard_cb = TensorboardCallback(
            self.directories_dict['tensorboard_dir'])

        # Logger callback
        logger_cb = LoggerCallback()

        # TQDM Callback
        val_tqdm_cb = ValProgressBar(self.num_val_iters)

        return [scheduler_cb, model_checkpoint_cb, tensorboard_cb, logger_cb, val_tqdm_cb]

    def train(self):
        self.model.fit(
            self.train_ds,
            epochs=self.train_params.num_epochs,
            initial_epoch=self.initial_epoch,
            validation_data=self.val_ds,
            validation_freq=self.train_params.eval_freq,
            callbacks=self.callbacks,
            verbose=1,
            workers=self.train_params.num_workers,
            use_multiprocessing=True
        )
        return
