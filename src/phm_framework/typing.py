import typing
import tensorflow as tf

ACTIVATIONS = ['relu', tf.keras.layers.LeakyReLU(alpha=0.1), 'tanh', 'sigmoid']

DISTANCES = ['l1', 'l2', 'cosine', 'product']

RNN_CELLS = [tf.keras.layers.GRUCell, tf.keras.layers.LSTMCell]

class NeedTypeConfiguration:
    pass

class CustomType:
    pass


class Activation(CustomType):
    @staticmethod
    def __getitem__(typ):
        return typing.Union[str, tf.keras.layers.Layer]

    def configure(self, task):
        pass

    def from_float(self, value: float):
        return ACTIVATIONS[round(value)]


class Distance(CustomType):
    @staticmethod
    def __getitem__(typ):
        return typing.Union[str]

    def configure(self, task):
        pass

    def from_float(self, value: float):
        return DISTANCES[round(value)]

class KernelSize(CustomType):
    dim1_max = 20

    @staticmethod
    def __getitem__(typ):
        return typing.Tuple[int, int]

    def configure(self, task):
        self.dim1_max = min(self.dim1_max, task['min_ts_len'])

    def from_float(self, value: float):
        dim0 = max(1, int(value))
        dim1 = max(1, int((value - int(value)) * self.dim1_max))

        return (dim0, dim1)


class RNNCell(CustomType):
    @staticmethod
    def __getitem__(typ):
        return typing.Union[str, tf.keras.layers.Layer]

    def configure(self, task):
        pass

    def from_float(self, value: float):
        return RNN_CELLS[round(value)]



def ensure_param(value, annotation, task):

    if issubclass(annotation, CustomType) and (isinstance(value, float) or isinstance(value, int)):
        annotation = annotation()
        annotation.configure(task)
        return annotation.from_float(value)

    elif annotation is int:
        return round(value)
    elif annotation is bool:
        return False if round(value) == 0 else True
    else:
        return value
