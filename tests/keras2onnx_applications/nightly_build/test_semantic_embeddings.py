# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from keras import regularizers
from keras.engine import Layer, InputSpec
from keras.utils import layer_utils, conv_utils
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_keras_and_ort, test_level_0
try:
    from keras.utils.conv_utils import normalize_data_format
except ImportError:
    from keras.backend import normalize_data_format
K = keras.backend

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Conv2D = keras.layers.Conv2D
Convolution2D = keras.layers.Convolution2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
GlobalMaxPooling2D = keras.layers.GlobalMaxPooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


class ChannelPadding(Layer):

    def __init__(self, padding=1, data_format=None, **kwargs):
        super(ChannelPadding, self).__init__(**kwargs)
        self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        axis = -1
        if input_shape[axis] is None:
            return input_shape
        else:
            length = input_shape[axis] + self.padding[0] + self.padding[1]
            if axis == 1:
                return input_shape[:1] + (length,) + input_shape[2:]
            else:
                return input_shape[:-1] + (length,)

    def call(self, inputs):
        pattern = [[0, 0] for i in range(len(inputs.shape))]
        axis = -1
        pattern[axis] = self.padding
        return K.tf.pad(inputs, pattern)

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(ChannelPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def simple_block(input_tensor, filters, prefix, kernel_size=3, stride=1,
                 regularizer=None, activation='relu', conv_shortcut=False, bn=True):
    bn_axis = 3
    conv_name_base = 'res' + prefix
    bn_name_base = 'bn' + prefix

    x = Conv2D(filters[1], kernel_size, padding='same', strides=(stride, stride),
               kernel_regularizer=regularizer,
               name=conv_name_base + 'x')(input_tensor)
    if bn:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'x')(x)
    x = Activation(activation)(x)

    x = Conv2D(filters[1], kernel_size, padding='same',
               kernel_regularizer=regularizer,
               name=conv_name_base + 'y')(x)
    if bn:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'y')(x)

    shortcut = input_tensor
    if (filters[0] != filters[1]) and conv_shortcut:
        shortcut = Conv2D(filters[1], (1, 1), strides=(stride, stride),
                          kernel_regularizer=regularizer,
                          name=conv_name_base + 'z')(shortcut)
        if bn:
            shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + 'z')(shortcut)
    else:
        if stride > 1:
            shortcut = AveragePooling2D((stride, stride), name='avg' + prefix)(shortcut)
        if filters[0] < filters[1]:
            shortcut = ChannelPadding(
                ((filters[1] - filters[0]) // 2, filters[1] - filters[0] - (filters[1] - filters[0]) // 2),
                name='pad' + prefix)(shortcut)

    x = keras.layers.add([x, shortcut])
    x = Activation(activation)(x)
    return x


def unit(input_tensor, filters, n, prefix, kernel_size=3, stride=1, **kwargs):
    x = simple_block(input_tensor, filters, prefix + '1', kernel_size=kernel_size, stride=stride, **kwargs)
    for i in range(1, n):
        x = simple_block(x, [filters[1], filters[1]], prefix + str(i + 1), kernel_size=kernel_size, **kwargs)
    return x


def SmallResNet(n=9, filters=[16, 32, 64],
                include_top=True, weights=None,
                input_tensor=None, input_shape=None,
                pooling='avg', regularizer=regularizers.l2(0.0002), activation='relu',
                top_activation='softmax',
                conv_shortcut=False, bn=True,
                classes=100, name=None):
    # Determine proper input shape
    if input_shape is None:
        input_shape = (32, 32, 3) if include_top and pooling is None else (None, None, 3)

    # Build network
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    bn_axis = 3

    x = Conv2D(filters[0], (3, 3), padding='same', name='conv0', kernel_regularizer=regularizer)(img_input)
    if bn:
        x = BatchNormalization(axis=bn_axis, name='bn0')(x)
    x = Activation(activation)(x)

    x = unit(x, [filters[0], filters[0]], n, '1-', kernel_size=3, stride=1, regularizer=regularizer,
             activation=activation, conv_shortcut=conv_shortcut, bn=bn)
    for i in range(1, len(filters)):
        x = unit(x, [filters[i - 1], filters[i]], n, str(i + 1) + '-', kernel_size=3, stride=2, regularizer=regularizer,
                 activation=activation, conv_shortcut=conv_shortcut, bn=bn)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)

    if include_top:
        x = Dense(classes, activation=top_activation, name='embedding' if top_activation is None else 'prob',
                  kernel_regularizer=regularizer)(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cifar-resnet{}'.format(2 * len(filters) * n) if name is None else name)

    return model


def PyramidNet(depth, alpha, bottleneck=True,
               include_top=True, weights=None,
               input_tensor=None, input_shape=None,
               pooling='avg', regularizer=regularizers.l2(0.0002),
               activation='relu', top_activation='softmax',
               classes=100, name=None):
    def shortcut(x, n, stride):
        if stride > 1:
            x = AveragePooling2D(stride)(x)
        input_channels = int(x.shape[-1])
        if input_channels < n:
            x = ChannelPadding((0, n - input_channels))(x)
        return x

    def basic_block(x, n, stride):
        s = BatchNormalization()(x)
        s = Conv2D(n, (3, 3), strides=stride, padding='same', kernel_initializer='glorot_normal',
                   kernel_regularizer=regularizer)(s)
        s = BatchNormalization()(s)
        s = Activation(activation)(s)
        s = Conv2D(n, (3, 3), padding='same', kernel_initializer='glorot_normal', kernel_regularizer=regularizer)(s)
        s = BatchNormalization()(s)
        return keras.layers.add([s, shortcut(x, n, stride)])

    def bottleneck_block(x, n, stride):
        s = BatchNormalization()(x)
        s = Conv2D(n, (1, 1), kernel_initializer='glorot_normal', kernel_regularizer=regularizer)(s)
        s = BatchNormalization()(s)
        s = Activation(activation)(s)
        s = Conv2D(n, (3, 3), strides=stride, padding='same', kernel_initializer='glorot_normal',
                   kernel_regularizer=regularizer)(s)
        s = BatchNormalization()(s)
        s = Activation(activation)(s)
        s = Conv2D(n * 4, (1, 1), kernel_initializer='glorot_normal', kernel_regularizer=regularizer)(s)
        s = BatchNormalization()(s)
        return keras.layers.add([s, shortcut(x, n * 4, stride)])

    def unit(x, features, count, stride):
        block = bottleneck_block if bottleneck else basic_block
        for i in range(count):
            x = block(x, features, stride)
        return x

    # Derived parameters
    n = (depth - 2) // 9 if bottleneck else (depth - 2) // 6
    channels = 16
    start_channel = 16
    add_channel = float(alpha) / (3 * n)

    # Determine proper input shape
    if input_shape is None:
        input_shape = (32, 32, 3) if include_top else (None, None, 3)

    # Build network
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    bn_axis = 3

    x = Conv2D(start_channel, (3, 3), padding='same', name='conv0', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizer)(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn0')(x)

    for b in range(3):
        start_channel += add_channel
        x = unit(x, round(start_channel), 1, 1 if b == 0 else 2)
        for i in range(1, n):
            start_channel += add_channel
            x = unit(x, round(start_channel), 1, 1)

    x = BatchNormalization(axis=bn_axis, name='bn4')(x)
    x = Activation(activation, name='act4')(x)

    # Final pooling
    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)

    # Top layer
    if include_top:
        x = Dense(classes, activation=top_activation, name='embedding' if top_activation is None else 'prob',
                  kernel_regularizer=regularizer)(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='pyramidnet-{}-{}'.format(depth, alpha) if name is None else name)

    return model


def PlainNet(output_dim,
             filters=[64, 64, 'ap', 128, 128, 128, 'ap', 256, 256, 256, 'ap', 512, 'gap', 'fc512'],
             activation='relu',
             regularizer=keras.regularizers.l2(0.0005),
             final_activation=None,
             input_shape=(None, None, 3),
             pool_size=(2, 2),
             name=None):

    prefix = '' if name is None else name + '_'

    flattened = False
    layers = [
        keras.layers.Conv2D(filters[0], (3, 3), padding='same', activation=activation, kernel_regularizer=regularizer,
                            input_shape=input_shape, name=prefix + 'conv1'),
        keras.layers.BatchNormalization(name=prefix + 'bn1')
    ]
    for i, f in enumerate(filters[1:], start=2):
        if f == 'mp':
            layers.append(keras.layers.MaxPooling2D(pool_size=pool_size, name='{}mp{}'.format(prefix, i)))
        elif f == 'ap':
            layers.append(keras.layers.AveragePooling2D(pool_size=pool_size, name='{}ap{}'.format(prefix, i)))
        elif f == 'gap':
            layers.append(keras.layers.GlobalAvgPool2D(name=prefix + 'avg_pool'))
            flattened = True
        elif isinstance(f, str) and f.startswith('fc'):
            if not flattened:
                layers.append(keras.layers.Flatten(name=prefix + 'flatten'))
                flattened = True
            layers.append(keras.layers.Dense(int(f[2:]), activation=activation, kernel_regularizer=regularizer,
                                             name='{}fc{}'.format(prefix, i)))
            layers.append(keras.layers.BatchNormalization(name='{}bn{}'.format(prefix, i)))
        else:
            layers.append(
                keras.layers.Conv2D(f, (3, 3), padding='same', activation=activation, kernel_regularizer=regularizer,
                                    name='{}conv{}'.format(prefix, i)))
            layers.append(keras.layers.BatchNormalization(name='{}bn{}'.format(prefix, i)))

    if not flattened:
        layers.append(keras.layers.Flatten(name=prefix + 'flatten'))
        flattened = True
    layers.append(keras.layers.Dense(output_dim, activation=final_activation,
                                     name=prefix + ('prob' if final_activation == 'softmax' else 'embedding')))

    return keras.models.Sequential(layers, name=name)


def initial_conv(input):
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(input)

    channel_axis = -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False)(init)

    channel_axis = -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    skip = Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      use_bias=False)(init)

    m = Add()([x, skip])

    return m


def conv_block(input, base, k=1, dropout=0.0):
    init = input

    channel_axis = -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False)(x)

    m = Add()([init, x])
    return m


def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, final_activation='softmax',
                                 verbose=1, name=None):
    channel_axis = -1

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    for block_index, base in enumerate([16, 32, 64]):

        x = expand_conv(x, base, k, strides=(2, 2) if block_index > 0 else (1, 1))

        for i in range(N - 1):
            x = conv_block(x, base, k, dropout)
            nb_conv += 2

        x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(nb_classes, activation=final_activation, name='prob' if final_activation == 'softmax' else 'embedding')(x)

    model = Model(ip, x, name=name)

    return model


# Models from https://github.com/cvjena/semantic-embeddings
class TestSemanticEmbeddings(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_PyramidNet(self):
        K.clear_session()
        keras_model = PyramidNet(272, 200)
        data = np.random.rand(1, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_SmallResNet(self):
        K.clear_session()
        keras_model = SmallResNet()
        data = np.random.rand(20, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_PlainNet(self):
        K.clear_session()
        keras_model = PlainNet(100)
        data = np.random.rand(200, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_wide_residual_network(self):
        K.clear_session()
        keras_model = create_wide_residual_network(input_dim=(32, 32, 3))
        data = np.random.rand(200, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
