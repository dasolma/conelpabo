from typing import Tuple

import numpy as np
import tensorflow as tf
from phm_framework import typing

EXTRA_CHANNEL = True

def create_model(input_shape, output,
                 block_size: int = 2,
                 nblocks: int = 2,
                 kernel_size: typing.KernelSize = (1, 10),
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
    dilation_rate = max(0, int(round(dilation_rate)))

    fc1 = int(fc1)
    fc2 = int(fc2)

    f1 = int(f1)
    f2 = int(f2)
    f3 = int(f3)
    ms_kernel_size = [f1, f2, f3]

    input_tensor = tf.keras.layers.Input(input_shape)
    x = input_tensor

    if fold_input:
        n = tf.shape(x)._inferred_value[-3]

        x = tf.transpose(x, perm=[0, 2, 1, 3])
        xshape = tf.shape(x)._inferred_value
        xshape[2] = n // 3
        xshape[3] = 3
        x = tf.reshape(x, [-1] + xshape[1:])


    for i, _ in enumerate(range(msblocks)):

        cblock = []
        for k in range(3):
            output_shape = x.shape
            f = ms_kernel_size[k]

            b = tf.keras.layers.Conv2D(filters, kernel_size=(f, 1), padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                       kernel_initializer='he_uniform',
                                       name='MSConv_%d%d_%d' % (i, k, f),
                                       dilation_rate=dilation_rate)(x)

            if batch_normalization:
                b = tf.keras.layers.BatchNormalization()(b)
            b = tf.keras.layers.Activation(conv_activation)(b)

            cblock.append(b)

        x = tf.keras.layers.Add()(cblock)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    for i, n_cnn in enumerate([block_size] * nblocks):
        for j in range(n_cnn):
            k1 = min(kernel_size[0], tf.shape(x)._inferred_value[-3])
            k2 = min(kernel_size[1], tf.shape(x)._inferred_value[-2])

            _filters = min(max_conv_filters, filters * 2 ** min(i, 2))
            x = tf.keras.layers.Conv2D(_filters, kernel_size=(k1, k2), padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                       kernel_initializer='he_uniform',
                                       name=f"C2D{i}.{j}_{k1}x{k2}",
                                       dilation_rate=dilation_rate)(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(conv_activation)(x)

        d1, d2 = tf.shape(x)._inferred_value[-3:-1]
        d1 = 1 if d1 < 2 else 2
        d2 = 1 if d2 < 2 else 2


        x = tf.keras.layers.MaxPooling2D((d1, d2))(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Flatten()(x)

    # FNN
    x = tf.keras.layers.Dense(fc1,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(fc2, tf.keras.layers.LeakyReLU(alpha=0.1), 
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                             name='features')(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_dim, activation=output, name='predictions')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x)

    return model
