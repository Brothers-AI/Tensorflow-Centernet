import sys

sys.path.append(".")
from tf_pipeline.utils.configs.parser import parse_args
from tf_pipeline.tf_utils.evaluater import TensorflowEvaluater

def main():

    args = parse_args()

    # Create the Evaluater
    trainer = TensorflowEvaluater(args.config)

    # Start the training
    trainer.evaluate()

    return

if __name__ == "__main__":
    main()