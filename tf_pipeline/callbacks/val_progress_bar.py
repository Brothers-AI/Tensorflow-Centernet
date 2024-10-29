from collections import defaultdict
import time

import tensorflow as tf

class ValProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, steps_per_epoch: int,
                 metrics_separator: str = " - ",
                 overall_bar_format: str = "{l_bar}{bar} {n_fmt}/{total_fmt} ETA: "
                 "{remaining}s,  {rate_fmt}{postfix}",
                 epoch_bar_format: str = "{n_fmt}/{total_fmt}{bar} ETA: "
                 "{remaining}s - {desc}",
                 metrics_format: str = "{name}: {value:0.4f}",
                 update_per_second: int = 10,
                 leave_epoch_progress: bool = True,
                 leave_overall_progress: bool = True,
                 show_epoch_progress: bool = True,
                 show_overall_progress: bool = True,):

        import tqdm
        version_message = "Please update your TQDM version to >= 4.36.1, "
        "you have version {}. To update, run !pip install -U tqdm"
        assert tqdm.__version__ >= "4.36.1", version_message.format(
            tqdm.__version__
        )
        from tqdm.auto import tqdm

        self.tqdm = tqdm

        self.metrics_separator = metrics_separator
        self.overall_bar_format = overall_bar_format
        self.epoch_bar_format = epoch_bar_format
        self.leave_epoch_progress = leave_epoch_progress
        self.leave_overall_progress = leave_overall_progress
        self.show_epoch_progress = show_epoch_progress
        self.show_overall_progress = show_overall_progress
        self.metrics_format = metrics_format

        # compute update interval (inverse of update per second)
        self.update_interval = 1 / update_per_second

        self.last_update_time = time.time()
        self.overall_progress_tqdm = None
        self.epoch_progress_tqdm = None
        self.is_training = False
        self.num_epochs = None
        self.logs = None
        self.steps_per_epoch = steps_per_epoch

        super(ValProgressBar, self).__init__()
    
    def _initialize_progbar(self, hook, epoch, logs=None):
        self.num_samples_seen = 0
        self.steps_to_update = 0
        self.steps_so_far = 0
        self.logs = defaultdict(float)
        self.num_epochs = self.params["epochs"]
        self.mode = "steps"
        self.total_steps = self.steps_per_epoch
        if hook == "test":
            if self.show_epoch_progress:
                self.epoch_progress_tqdm = self.tqdm(
                    total=self.total_steps,
                    desc="Evaluating",
                    bar_format=self.epoch_bar_format,
                    leave=self.leave_epoch_progress,
                    dynamic_ncols=True,
                    unit=self.mode,
                )
        else:
            raise NotImplementedError(f'Not Implemented for hook {hook}')
    
    def _clean_up_progbar(self, hook, logs):
        if hook == "test":
            metrics = self.format_metrics(logs, self.num_samples_seen)
        else:
            raise NotImplementedError(f'Not Implemented for hook {hook}')

        if self.show_epoch_progress:
            self.epoch_progress_tqdm.desc = metrics
            # set miniters and mininterval to 0 so last update displays
            self.epoch_progress_tqdm.miniters = 0
            self.epoch_progress_tqdm.mininterval = 0
            # update the rest of the steps in epoch progress bar
            self.epoch_progress_tqdm.update(
                self.total_steps - self.epoch_progress_tqdm.n
            )
            self.epoch_progress_tqdm.close()
    
    def _update_progbar(self, logs):
        if self.mode == "samples":
            batch_size = logs["size"]
        else:
            batch_size = 1

        self.num_samples_seen += batch_size
        self.steps_to_update += 1
        self.steps_so_far += 1

        if self.steps_so_far <= self.total_steps:
            for metric, value in logs.items():
                self.logs[metric] += value * batch_size

            now = time.time()
            time_diff = now - self.last_update_time
            if self.show_epoch_progress and time_diff >= self.update_interval:

                # update the epoch progress bar
                metrics = self.format_metrics(self.logs, self.num_samples_seen)
                self.epoch_progress_tqdm.desc = metrics
                self.epoch_progress_tqdm.update(self.steps_to_update)

                # reset steps to update
                self.steps_to_update = 0

                # update timestamp for last update
                self.last_update_time = now
    
    def on_test_begin(self, logs={}):
        if not self.is_training:
            self._initialize_progbar("test", None, logs)

    def on_test_end(self, logs={}):
        if not self.is_training:
            self._clean_up_progbar("test", self.logs)
    
    def on_test_batch_end(self, batch, logs={}):
        if not self.is_training:
            self._update_progbar(logs)
    
    def format_metrics(self, logs={}, factor=1):
        """Format metrics in logs into a string.

        Args:
            logs: dictionary of metrics and their values. Defaults to
                empty dictionary.
            factor (int): The factor we want to divide the metrics in logs
                by, useful when we are computing the logs after each batch.
                Defaults to 1.

        Returns:
            metrics_string: a string displaying metrics using the given
            formators passed in through the constructor.
        """

        metric_value_pairs = []
        for key, value in logs.items():
            if key in ["batch", "size"]:
                continue
            pair = self.metrics_format.format(name=key, value=value / factor)
            metric_value_pairs.append(pair)
        metrics_string = self.metrics_separator.join(metric_value_pairs)
        return metrics_string
    
    def get_config(self):
        config = {
            "metrics_separator": self.metrics_separator,
            "overall_bar_format": self.overall_bar_format,
            "epoch_bar_format": self.epoch_bar_format,
            "leave_epoch_progress": self.leave_epoch_progress,
            "leave_overall_progress": self.leave_overall_progress,
            "show_epoch_progress": self.show_epoch_progress,
            "show_overall_progress": self.show_overall_progress,
        }

        base_config = super().get_config()
        return {**base_config, **config}


