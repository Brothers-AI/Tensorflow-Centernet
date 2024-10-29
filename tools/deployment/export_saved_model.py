import os
import sys
import argparse
import logging

sys.path.append(".")

from tf_pipeline.models.create import create_tf_model
from tf_pipeline.utils.configs.config import Config

def parse_args():
    parser = argparse.ArgumentParser("Export to Saved Model")
    parser.add_argument("--config", required=True, type=str,
                        help='Config file')
    parser.add_argument("--num_classes", type=int, default=7,
                        help='Num of classes in head')
    parser.add_argument("--out_name", type=str, default='peleenet',
                        help='Out folder name of the saved model')
    return parser.parse_args()

def create_and_load_model(config: Config, logger: logging.Logger, num_classes: int):
    # Create the TF Model
    model = create_tf_model(config.tf_module, config.model_config, [
        *config.model_resolution, 3], num_classes=num_classes)
    if config.ckpt_path:
        logger.info(
            f"Loading from checkpoint {config.ckpt_path}")
        model.load_weights(config.ckpt_path).expect_partial()
    else:
        logger.error(
            f"Checkpoint path not given, Please give the checkpoint path")
        exit(-1)
    model.compile()
    model.summary()
    return model

def main():

    args = parse_args()

    config_path = args.config
    num_classes = args.num_classes
    out_name = args.out_name

    # get config
    config = Config.parse_file(config_path)

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger()

    logger.info(f"Config file -> {config_path}")
    logger.info(f"Config file contents\n{config}")

    # Create and load model
    model = create_and_load_model(config, logger, num_classes)

    # Save the model
    logger.info(f"Saving model to {out_name}")
    model.save(out_name)
    logger.info(f"Model saved at {out_name}")
    return

if __name__ == "__main__":
    main()