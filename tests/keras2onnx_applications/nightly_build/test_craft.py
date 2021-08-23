# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras, is_tf_keras
from keras.applications.vgg16 import VGG16
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_onnx_runtime
import tensorflow as tf
from keras.utils import conv_utils
from keras.engine import Layer, InputSpec
import keras.backend as K
from onnxconverter_common.onnx_ex import get_maximum_opset_supported


Activation = keras.layers.Activation
BatchNormalization = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
concatenate = keras.layers.concatenate
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def _collect_input_shape(input_tensors):
    input_tensors = _to_list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(K.int_shape(x))
        except Exception as e:
            print(e)
            shapes.append(None)
    if len(shapes) == 1:
        return shapes[0]

    return shapes


def _permute_dimensions(x, pattern):
    return tf.transpose(x, perm=pattern)


def _resie_image(x, target_layer, target_shape, data_format):
    if data_format == 'channels_first':
        new_shape = tf.shape(target_layer)[2:]
        x = _permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x = _permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, target_shape[2], target_shape[3]))
        return x
    elif data_format == 'channels_last':
        new_shape = tf.shape(target_layer)[1:3]
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x.set_shape((None, target_shape[1], target_shape[2], None))
        return x
    else:
        raise ValueError('Unknown data_format: ' + str(data_format))


class Interpolate(Layer):
    def __init__(self, target_layer, data_format=None, **kwargs):
        super(Interpolate, self).__init__(**kwargs)
        self.target_layer = target_layer
        self.target_shape = _collect_input_shape(target_layer)
        # self.data_format = conv_utils.normalize_data_format(data_format)
        self.data_format = K.normalize_data_format(data_format)

        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.target_shape[2]
            width = self.target_shape[3]
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.target_shape[1]
            width = self.target_shape[2]
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs, **kwargs):
        return _resie_image(inputs, self.target_layer, self.target_shape, self.data_format)


def up_conv(input_tensor, filters):
    x = Conv2D(filters[0], kernel_size=1)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters[1], kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def conv_cls(input_tensor, num_class):
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=num_class, padding='same', activation='sigmoid')(x)
    return x


def VGG16_UNet(weights=None, input_tensor=None, pooling=None):
    vgg16 = VGG16(include_top=False, weights=weights, input_tensor=input_tensor, pooling=pooling)

    source = vgg16.get_layer('block5_conv3').output
    x = MaxPooling2D(3, strides=1, padding='same', name='block5_pool')(source)
    x = Conv2D(1024, kernel_size=3, padding='same', dilation_rate=6)(x)
    x = Conv2D(1024, kernel_size=1)(x)

    x = Interpolate(target_layer=source, name='resize_1')(x)
    x = concatenate([x, source])
    x = up_conv(x, [512, 256])

    source = vgg16.get_layer('block4_conv3').output
    x = Interpolate(target_layer=source, name='resize_2')(x)
    x = concatenate([x, source])
    x = up_conv(x, [256, 128])

    source = vgg16.get_layer('block3_conv3').output
    x = Interpolate(target_layer=source, name='resize_3')(x)
    x = concatenate([x, source])
    x = up_conv(x, [128, 64])

    source = vgg16.get_layer('block2_conv2').output
    x = Interpolate(target_layer=source, name='resize_4')(x)
    x = concatenate([x, source])
    feature = up_conv(x, [64, 32])

    x = conv_cls(feature, 2)

    region_score = Lambda(lambda layer: layer[:, :, :, 0])(x)
    affinity_score = Lambda(lambda layer: layer[:, :, :, 1])(x)

    return region_score, affinity_score

# From https://github.com/RubanSeven/CRAFT_keras/blob/master/net/vgg16.py
class TestCRAFT(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(get_maximum_opset_supported() < 10,
                     "Need Upsample 10+ support.")
    def test_CRAFT(self):
        # input_image = Input(shape=(None, None, 3)) -- Need fixed input shape
        input_image = Input(shape=(512, 512, 3))
        region, affinity = VGG16_UNet(input_tensor=input_image, weights=None)
        keras_model = Model(input_image, [region, affinity], name='vgg16_unet')
        x = np.random.rand(1, 512, 512, 3).astype(np.float32)
        expected = keras_model.predict(x)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
