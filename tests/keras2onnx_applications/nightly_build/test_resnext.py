# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from mock_keras2onnx.proto.tfcompat import is_tf2
if is_tf2:
    def l2(weight_decay):
        # old keras layer expects a tuple but tf keras wants a single value
        return keras.regularizers.l2(weight_decay[0])
else:
    from keras.regularizers import l2
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')
from test_utils import test_level_0, run_image
K = keras.backend

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
add = keras.layers.add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Conv2D = keras.layers.Conv2D
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


def initial_conv_block(input, weight_decay=5e-4):
    channel_axis = -1

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x


def initial_conv_block_inception(input_tensor, weight_decay=5e-4):
    channel_axis = -1

    x = Conv2D(64, (7, 7), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), strides=(2, 2))(input_tensor)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    return x


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input
    channel_axis = -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    '''
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
            if K.image_data_format() == 'channels_last' else
            lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        group_list.append(x)
    '''
    x = Lambda(lambda z: z[:, :, :, 0 * grouped_channels:(0 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 1 * grouped_channels:(1 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 2 * grouped_channels:(2 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 3 * grouped_channels:(3 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 4 * grouped_channels:(4 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 5 * grouped_channels:(5 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 6 * grouped_channels:(6 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 7 * grouped_channels:(7 + 1) * grouped_channels])(input)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation('relu')(x)

    return x


def bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input

    grouped_channels = int(filters / cardinality)
    channel_axis = -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if init.shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = add([init, x])
    x = Activation('relu')(x)

    return x


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def __grouped_se_convolution_block(input_tensor, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input_tensor
    channel_axis = -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = LeakyReLU()(x)
        return x

    '''
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda _z: _z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input_tensor)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)
    '''
    x = Lambda(lambda z: z[:, :, :, 0 * grouped_channels:(0 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 1 * grouped_channels:(1 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 2 * grouped_channels:(2 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 3 * grouped_channels:(3 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 4 * grouped_channels:(4 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 5 * grouped_channels:(5 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 6 * grouped_channels:(6 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)
    x = Lambda(lambda z: z[:, :, :, 7 * grouped_channels:(7 + 1) * grouped_channels])(input_tensor)
    x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = LeakyReLU()(x)

    return x


def se_bottleneck_block(input_tensor, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input_tensor

    grouped_channels = int(filters / cardinality)
    channel_axis = -1

    if init.shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU()(x)

    x = __grouped_se_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    x = add([init, x])
    x = LeakyReLU()(x)

    return x


def create_res_next(__initial_conv_block, __bottleneck_block,
                    nb_classes, img_input, include_top, depth=29, cardinality=8, width=4,
                    weight_decay=5e-4, pooling=None):
    if type(depth) is list or type(depth) is tuple:
        # If a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        # Otherwise, default to 3 blocks each of default number of group convolution blocks
        N = [(depth - 2) // 9 for _ in range(3)]

    filters = cardinality * width
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters

    x = __initial_conv_block(img_input, weight_decay)

    # block 1 (no pooling)
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)

    N = N[1:]  # remove the first block from block definition list
    filters_list = filters_list[1:]  # remove the first filter from the filter list

    # block 2 to N
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=2,
                                       weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=1,
                                       weight_decay=weight_decay)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal', activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x


class TestResNext(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    # Model from https://github.com/titu1994/Keras-ResNeXt
    @unittest.skipIf(not is_tf2,
                     "This is a tf2 model.")
    def test_ResNext(self):
        K.clear_session()
        input_shape = (112, 112, 3)
        depth = 29
        cardinality = 8
        width = 64
        weight_decay = 5e-4,
        include_top = True
        pooling = None
        classes = 10

        img_input = keras.layers.Input(shape=input_shape)
        x = create_res_next(initial_conv_block, bottleneck_block,
                            classes, img_input, include_top, depth, cardinality, width,
                            weight_decay, pooling)
        inputs = img_input

        keras_model = Model(inputs, x, name='resnext')
        res = run_image(keras_model, self.model_files, img_path, atol=5e-3, target_size=112)
        self.assertTrue(*res)

    # Model from https://github.com/titu1994/keras-squeeze-excite-network
    @unittest.skipIf(not is_tf2,
                     "This is a tf2 model.")
    def test_SEResNext(self):
        K.clear_session()
        input_shape = (112, 112, 3)
        depth = 29
        cardinality = 8
        width = 64
        weight_decay = 5e-4,
        include_top = True
        pooling = None
        classes = 10

        img_input = keras.layers.Input(shape=input_shape)
        x = create_res_next(initial_conv_block_inception, se_bottleneck_block,
                            classes, img_input, include_top, depth, cardinality, width,
                            weight_decay, pooling)
        inputs = img_input

        keras_model = Model(inputs, x, name='se_resnext')
        res = run_image(keras_model, self.model_files, img_path, atol=5e-3, target_size=112)
        self.assertTrue(*res)


if __name__ == "__main__":
    unittest.main()
