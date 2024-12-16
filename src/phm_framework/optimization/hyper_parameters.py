import phm_framework as phmf
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import tensorflow as tf
import os
import logging
from phm_framework import scoring
from phm_framework.utils import flat_dict

logging.basicConfig(level=logging.INFO)
logging.info("Working dir: " + os.getcwd())

OUTPUT = [
    {
        'field': 'target',
        'value': 'rul',
        'output': 'relu'
    },

]

LOSS_METRICS = [
    {
        'field': 'type',
        'value': 'regression',
        'output': ['mse',
                   lambda task: tf.keras.metrics.MeanSquaredError(name='mse'),
                   lambda task: tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                   lambda task: tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                   lambda task: tf.keras.metrics.MeanAbsoluteError(name="mae"),
                   lambda task: scoring.SMAPE(name="smape"),
                   lambda task: scoring.NASAScore(name="nasa_score"),
                   ]
    },

]

OUTPUT_DIM = [
    {
        'field': 'type',
        'value': 'regression',
        'output': 1
    }
]



RANGES = {
    # nlp
    'nhideen_layers': (1, 7),
    'activation': (-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),

    # mscnn
    'kernel_size': (lambda task: min(3, len(task['features'])) + .0,
                    lambda task: min(15, len(task['features'])) + 0.1),
    'msblocks': (-0.49, 3.49),
    'block_size': (-0.49, 5.49),
    'f1': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(10, task['min_ts_len'])),
    'f2': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(10, task['min_ts_len'])),
    'f3': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(10, task['min_ts_len'])),
    'filters': (16, 64),
    'conv_activation': (-0.49, 1.49), #(-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),

    'dilation_rate': (0.51, 10.49),


    # rnn
    'cell_type': (-0.49, 1.49),
    'rnn_units': (32, 256),
    'bidirectional': (0, 1),

    # general
    'nblocks': (0.51, 4.49),
    'fc1': (16, 64),
    'fc2': (16, 64),
    'dense_activation':  (-0.49, 1.49), #(-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),
    'batch_normalization': (0, 1),

    # regularization
    'dropout': (0, 0.9),
    'l1': (0, 0.00001),
    'l2': (0, 0.00001),


}

class DummyPreprocess():

    def fit(self, X):
        pass

    def transform(self, X):
        return X

PREPROCESS = {
    None: DummyPreprocess,
    'norm': MinMaxScaler,
    'std': StandardScaler,
}

POINTS_TO_EVALUATE = {

}

DATASET_FIXED_PARAMS = {

}


def remove_fixed_params_ds(params, dataset_name):

    if dataset_name in DATASET_FIXED_PARAMS:
        fixed_params = flat_dict(DATASET_FIXED_PARAMS[dataset_name].copy())

        for key in fixed_params.keys():

            if key in params:
                del params[key]

    return params

def remove_fixed_params(params, fixed_params):

    fixed_params = flat_dict(fixed_params.copy())

    for key in fixed_params.keys():

        if key in params:
            del params[key]

    return params

def update_dict(d1, d2, task):
    keys = list(d2.keys())

    if len(keys) == 0:
        return d1

    key = keys[0]
    value = d2[key]
    if key in d1:

        if isinstance(value, dict):
            d1[key] = update_dict(d1[key], value, task)

        elif callable(value):
            d1[key] = value(task)
        else:
            d1[key] = value

    else:
        if callable(value) and key != 'extra_callbacks':
            d1[key] = value(task)
        else:
            d1[key] = value

    d2 = d2.copy()
    del d2[key]

    value = d1[key]
    del d1[key]

    r = update_dict(d1, d2, task)
    r[key] = value

    return r


def get_points_to_evaluate(optimization_config, task):
    net = optimization_config['model']['net']
    dataset_name = optimization_config['data']['dataset_name']

    if net in POINTS_TO_EVALUATE.keys():
        result = [update_dict(optimization_config, remove_fixed_params_ds(point, dataset_name), task)
                  for point in POINTS_TO_EVALUATE[net]]

        logging.info(f"Settting points to evaluate: {result}")

        return result

    return []


def get_config(task, rules):
    for rule in rules:
        if task[rule['field']] == rule['value']:
            o = rule['output']

            o = compute_value(o, task)

            return o


def compute_value(o, task):
    if isinstance(o, list):
        o = [e(task) if callable(e) else e for e in o]
    elif callable(o):
        o = o(task)
    return o


def get_loss(task):
    return get_config(task, LOSS_METRICS)


def get_output(task):
    return get_config(task, OUTPUT)


def get_output_dim(task):
    return get_config(task, OUTPUT_DIM)


def compute_ranges(task):
    return {k: (compute_value(v1, task), compute_value(v2, task)) for k, (v1, v2) in RANGES.items()}
