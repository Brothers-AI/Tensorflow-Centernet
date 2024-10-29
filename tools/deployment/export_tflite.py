import os
import sys
import argparse
import logging

import tensorflow as tf

def parse_args():

    parser = argparse.ArgumentParser("Export to tflite")
    parser.add_argument("--saved_model_path", required=True, type=str,
                        help='Path of the saved model')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch size of the tflite model')
    parser.add_argument("--tflite_model_path", type=str, default='peleenet_nas.tflite',
                        help='TFlite model path')
    parser.add_argument("--input_shape", type=str, default='512,768,3',
                        help='Input shape of the model (H, W, C)')
    return parser.parse_args()

def main():

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger()

    args = parse_args()
    saved_model_path = args.saved_model_path
    batch_size = args.batch_size
    tflite_model_path = args.tflite_model_path
    input_shape = args.input_shape

    input_shape = list(map(int, input_shape.split(",")))
    input_shape = (batch_size, *input_shape)

    logger.info(f"Saved Model path: {saved_model_path}")
    logger.info(f"Input Shape (N, H, W, C): {input_shape}")
    logger.info(f"TFLite Model path: {tflite_model_path}")

    # Load Saved Model
    logger.info(f"Loading saved model from {saved_model_path}")
    model = tf.keras.models.load_model(saved_model_path)
    logger.info(f"Loaded saved model from {saved_model_path} successfully.")

    model.summary()

    # Convert Keras model to ConcreteFunction
    logger.info(f"Converting saved model to Concrete Function")
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(input_shape, model.inputs[0].dtype))
    logger.info(f"Converted saved model to Concrete Function")

    # Convert the model
    logger.info('Converting to tflite')
    converter = tf.lite.TFLiteConverter.from_concrete_functions([full_model])
    tflite_model = converter.convert()
    logger.info('Converted to tflite')

    # Save the model.
    with open(f'{tflite_model_path}', 'wb') as f:
        f.write(tflite_model)
    logger.info(f'Saved tflite model to: {tflite_model_path}')

    return

if __name__ == "__main__":
    main()