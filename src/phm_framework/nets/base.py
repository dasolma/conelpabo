import tensorflow as tf

class ModelWrapper(tf.keras.Model):

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_model = model

    @property
    def metrics_names(self):
        return self.base_model.metrics_names

    @property
    def metrics(self):
        return self.base_model.metrics

    def compile(self, *args, **kwargs):
        return self.base_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.base_model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.base_model.evaluate(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.base_model.summary(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.base_model.save(*args, **kwargs)

    def call(self, *args, **kwargs):
        return self.base_model.call(*args, **kwargs)