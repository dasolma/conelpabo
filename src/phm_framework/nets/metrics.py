import logging
from typeguard import typechecked
import datetime
import time
import tensorflow as tf

class AdditionalRULValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets):
        """
        :param validation_sets:
        a list of 2-tuples (validation_gen, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalRULValidationSets, self).__init__()

        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2]:
                raise ValueError()
        self.epoch = []
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_gen, name in self.validation_sets:

            results = self.model.evaluate(validation_gen,
                                          verbose=0)

            for metric, result in zip(self.model.metrics_names, results):
                valuename = name + '_' + metric

                self.history.setdefault(valuename, []).append(result)

                logs[valuename] = result


@tf.keras.utils.register_keras_serializable(package="phm_framework")
class TimeStopping(tf.keras.callbacks.Callback):
    """Stop training when a specified amount of time has passed.

    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
        verbose: verbosity mode. Defaults to 0.
    """

    @typechecked
    def __init__(self, seconds: int = 86400, verbose: int = 0):
        super().__init__()

        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds

    def on_epoch_end(self, epoch, logs={}):
        if time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = "Timed stopping at epoch {} after training for {}".format(
                self.stopped_epoch + 1, formatted_time
            )
            print(msg)

    def get_config(self):
        config = {
            "seconds": self.seconds,
            "verbose": self.verbose,
        }

        base_config = super().get_config()
        return {**base_config, **config}

