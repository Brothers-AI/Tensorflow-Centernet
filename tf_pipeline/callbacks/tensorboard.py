import tensorflow as tf

class TensorboardCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super(TensorboardCallback, self).__init__(log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({"lr": tf.keras.backend.get_value(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
    
    def on_batch_end(self, batch, logs=None):
        logs.update({"lr": tf.keras.backend.get_value(self.model.optimizer.lr)})
        super().on_batch_end(batch, logs)
    
