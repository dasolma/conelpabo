import inspect

from . import rnn
from . import mscnn
from . import mscnn1d
from .. import typing


def get_model_params(config, func, task):
    config = config.copy()

    params = inspect.signature(func).parameters
    param_names = list(params.keys())

    #config = {k: typing.ensure_param(v, params[k].annotation) for k, v in config.items() if k in param_names}
    config = {k: typing.ensure_param(config[k], params[k].annotation, task) if k in config else params[k].default
              for k in param_names if k in param_names}
    return config
