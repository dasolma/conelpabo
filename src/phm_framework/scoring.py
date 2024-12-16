import tensorflow as tf


class NASAScore(tf.keras.metrics.Metric):

    def __init__(self, name='NASA_score', **kwargs):
        super(NASAScore, self).__init__(name=name, **kwargs)
        self.scores = self.add_weight(name='sc', initializer='zeros')
        self.total_samples = self.add_weight(name='ts', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_samples.assign_add(tf.shape(y_true)[0])

        # residual = y_true - tf.math.maximum(y_pred, [0])
        residual = y_true - y_pred
        ueb = tf.cast(tf.less_equal(residual, [0]), tf.float32)
        uev = tf.math.exp((1 / 13) * tf.math.abs(residual * ueb)) - 1
        ovv = tf.math.exp((1 / 10) * tf.math.abs(residual * (1 - ueb))) - 1

        self.scores.assign_add(tf.reduce_sum(uev) + tf.reduce_sum(ovv))

    def result(self):
        return self.scores / tf.cast(self.total_samples, tf.float32)

    def reset_states(self):
        self.scores.assign(0)
        self.total_samples.assign(0)

class SMAPE(tf.keras.metrics.Metric):

    def __init__(self, name='smape', **kwargs):
        super(SMAPE, self).__init__(name=name, **kwargs)
        self.scores = self.add_weight(name='sc', initializer='zeros')
        self.total_samples = self.add_weight(name='ts', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_samples.assign_add(tf.shape(y_true)[0])

        # residual = y_true - tf.math.maximum(y_pred, [0])
        residual = y_true - y_pred

        _scores = tf.math.abs(residual) / (((tf.math.abs(y_true) + tf.math.abs(y_pred)) * 0.5) + 1e-6)
        self.scores.assign_add(tf.reduce_sum(_scores))

    def result(self):
        return self.scores / tf.cast(self.total_samples, tf.float32)

    def reset_states(self):
        self.scores.assign(0)
        self.total_samples.assign(0)


def loss_score(y_true, y_pred):
    r = (y_true - y_pred)
    return tf.reduce_mean(r ** 2) + tf.math.exp(r / 5)