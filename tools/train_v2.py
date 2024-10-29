import sys

sys.path.append(".")
from tf_pipeline.utils.configs.parser import parse_args
from tf_pipeline.tf_utils.trainer import TensorflowTrainer

def main():

    args = parse_args()

    # Create the trainer
    trainer = TensorflowTrainer(args.config)

    # Start the training
    trainer.train()

    return

if __name__ == "__main__":
    main()