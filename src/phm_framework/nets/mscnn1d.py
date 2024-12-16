from typing import Tuple

import numpy as np
import tensorflow as tf
from phm_framework import typing

EXTRA_CHANNEL = False

def create_model(input_shape, output,
                 block_size: int = 2,
                 nblocks: int = 2,
                 kernel_size: int = 32,
                 l1: float = 1e-5, l2: float = 1e-4,
                 msblocks: int = 2,
                 f1: int = 10,
                 f2: int = 15,
                 f3: int = 20,
                 dropout: float = 0.5,
                 filters: int = 64,
                 fc1: int = 256,
                 fc2: int = 128,
                 conv_activation: typing.Activation = 'relu',
                 dense_activation: typing.Activation = 'relu',
                 dilation_rate: int = 1,
                 batch_normalization: bool = True,
                 fold_input: bool = False,
                 output_dim: int = 1,
                 max_conv_filters=128):


    block_size = int(round(block_size))
    nblocks = int(round(nblocks))
    fc1 = int(round(fc1))
    fc2 = int(round(fc2))
    dilation_rate = max(1, int(round(dilation_rate)))

    fc1 = int(fc1)
    fc2 = int(fc2)

    f1 = int(f1)
    f2 = int(f2)
    f3 = int(f3)
    ms_kernel_size = [f1, f2, f3]

    input_tensor = tf.keras.layers.Input(input_shape)
    x = input_tensor

    #x = tf.transpose(x, [0, 2, 1])
    x = tf.keras.layers.Permute((2, 1))(x)

    cact_name = conv_activation if isinstance(conv_activation, str) else conv_activation.__class__.__name__.lower()
    dact_name = dense_activation if isinstance(dense_activation, str) else dense_activation.__class__.__name__.lower()

    for i, _ in enumerate(range(msblocks)):

        cblock = []
        for k in range(3):
            output_shape = x.shape
            f = ms_kernel_size[k]

            b = tf.keras.layers.Conv1D(filters, kernel_size=f, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                       kernel_initializer='he_uniform',
                                       name='MSConv_%d%d_%d' % (i, k, f),
                                       dilation_rate=dilation_rate)(x)

            if batch_normalization:
                b = tf.keras.layers.BatchNormalization()(b)


            b = tf.keras.layers.Activation(conv_activation, name=f"MSC_{i}{k}_{f}_act_{cact_name}")(b)

            cblock.append(b)

        x = tf.keras.layers.Add()(cblock)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    for i, n_cnn in enumerate([block_size] * nblocks):
        for j in range(n_cnn):
            _filters = min(max_conv_filters, filters * 2 ** min(i, 2))
            x = tf.keras.layers.Conv1D(_filters, kernel_size=kernel_size, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                       kernel_initializer='he_uniform',
                                       name=f"C2D{i}.{j}_{kernel_size}",
                                       dilation_rate=dilation_rate)(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation(conv_activation, name=f"C2D{i}.{j}_act_{cact_name}")(x)

        d1 = tf.shape(x)._inferred_value[-2]
        d1 = 1 if d1 < 2 else 2

        x = tf.keras.layers.MaxPooling1D(d1)(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Flatten()(x)

    # FNN
    x = tf.keras.layers.Dense(fc1, name="Dense1",
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation, name=f"Dense1_act_{dact_name}")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(fc2, tf.keras.layers.LeakyReLU(alpha=0.1), name="Dense2",
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation, name='features')(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_dim, activation=output, name='predictions')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x)

    return model
