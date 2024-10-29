import os
import sys
import time
import logging

import tensorflow as tf

sys.path.append(".")

from tf_pipeline.utils.registry.datasets import DATASETS
from tf_pipeline.datasets import DefaultDataset
from tf_pipeline.utils.configs.parser import parse_args
from tf_pipeline.utils.configs.config import Config
from tf_pipeline.models.create import create_tf_model
from tf_pipeline.callbacks import LoggerCallback, TensorboardCallback


def main():

    # Parse the arguments
    args = parse_args()

    # Parse the config file
    config = Config.parse_file(args.config)

    # Save directory
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    exp_save_dir = os.path.join(config.save_dir, config.exp_name)
    logs_dir = os.path.join(exp_save_dir, "logs")
    log_filename = f"logfile_{time_str}.log"
    tensorboard_logs_dir = os.path.join(exp_save_dir, "tensorboard_logs")
    checkpoints_dir = os.path.join(exp_save_dir, "checkpoints")

    # Create the directories
    os.makedirs(exp_save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, log_filename)),
            logging.StreamHandler(sys.stdout)
        ]
    )

    LOG = logging.getLogger()

    LOG.info(f"Config file -> {args.config}")
    LOG.info(f"Config file contents\n{config}")

    # Copy the config to exp_dir
    os.system(f"cp {args.config} {exp_save_dir}")

    # TODO: uncomment when the callbacks added
    """
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    """

    # Get the train params
    train_params = config.train_params
    train_dataset_name, train_dataset_kwargs = train_params.dataset_config()

    # Create the dataset
    train_dataset: DefaultDataset = DATASETS.get(train_dataset_name)(model_resolution=config.model_resolution,
                                                                     split="train", init_lr=train_params.lr,
                                                                     **train_dataset_kwargs)
    train_ds = tf.data.Dataset.from_generator(train_dataset, output_types=train_dataset.output_types,
                                              output_shapes=train_dataset.output_shapes)
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.batch(train_params.batch_size)

    # Get the val params
    val_params = config.val_params
    val_dataset_name, val_dataset_kwargs = val_params.dataset_config()

    # Create the val dataset
    val_dataset: DefaultDataset = DATASETS.get(val_dataset_name)(model_resolution=config.model_resolution,
                                                                 split="val", **val_dataset_kwargs)
    val_ds = tf.data.Dataset.from_generator(val_dataset, output_types=val_dataset.output_types,
                                            output_shapes=val_dataset.output_shapes)
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    val_ds = val_ds.with_options(options)
    val_ds = val_ds.batch(1)

    # Inital Epoch
    initial_epoch = 0

    # Create Scheduler
    scheduler_cb = tf.keras.callbacks.LearningRateScheduler(train_dataset.scheduler)
    if config.resume:
        if isinstance(config.resume, bool):
            raise TypeError("resume should be given with the checkpoint number")
        optimizer = tf.keras.optimizers.Adam(train_dataset.scheduler(config.resume))
        initial_epoch = config.resume
    else:
        optimizer = tf.keras.optimizers.Adam(train_dataset.scheduler(0))

    # For Multi GPU
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Create the TF Model
        model = create_tf_model(config.tf_module, config.model_config, [
            *config.model_resolution, 3], num_classes=train_dataset.num_classes)
        
        if config.resume:
            if config.ckpt_path:
                LOG.info(f"Resuming from checkpoint {config.ckpt_path}")
                model.load_weights(config.ckpt_path).expect_partial()
            else:
                LOG.error(f"Checkpoint path not given, Not loading previous checkpoint weights")

        # Compile the model
        model.compile(optimizer=optimizer)
        model.summary()
    
    # Model Checkpoint callback
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoints_dir, "model_{epoch}"),
        save_freq="epoch",
        verbose=1,
        save_best_only=False,
        save_weights_only=True)
    
    callbacks = [scheduler_cb, model_checkpoint_cb]

    # Start the training
    model.fit(
        train_ds,
        epochs=train_params.num_epochs,
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        validation_freq=train_params.eval_freq,
        callbacks=callbacks,
        verbose=1,
    )
    return


if __name__ == "__main__":
    main()
