import logging

import tensorflow as tf

class LoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LoggerCallback, self).__init__()
        self.logger = logging.getLogger()

    def on_epoch_end(self, epoch, logs=None):
        msg = f"Epoch: {epoch + 1}: "
        for key, value in logs.items():
            msg += f"{key}: {value}, "
        self.logger.info(msg)