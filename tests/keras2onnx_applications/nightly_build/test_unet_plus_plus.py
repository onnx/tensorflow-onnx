# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import numpy as np
import onnxruntime
from os.path import dirname, abspath
from mock_keras2onnx.proto import keras, is_keras_older_than
from keras.applications.vgg16 import VGG16
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from distutils.version import StrictVersion

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_image
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')


Input = keras.layers.Input
Activation = keras.layers.Activation
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
MaxPooling2D = keras.layers.MaxPooling2D
BatchNormalization = keras.layers.BatchNormalization

Model = keras.models.Model

def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name

def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)
        if skip is not None:
            x = Concatenate()([x, skip])
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer


def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)
        if skip is not None:
            x = Concatenate()([x, skip])
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer


def get_layer_number(model, layer_name):
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)
    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


# From https://github.com/MrGiovanni/UNetPlusPlus
class TestUnetPlusPlus(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(StrictVersion(onnxruntime.__version__.split('-')[0]) < StrictVersion('1.7.0'),
                     "ConvTranspose stride > 1 is fixed in onnxruntime 1.7.0.")
    def test_unet_plus_plus(self):
        backbone_name = 'vgg16'
        input_shape = (None, None, 3)
        input_tensor = None
        encoder_weights = None#'imagenet'

        backbone = VGG16(input_shape=input_shape,
                         input_tensor=input_tensor,
                         weights=encoder_weights,
                         include_top=False)

        input = backbone.input
        x = backbone.output
        block_type = 'transpose'

        if block_type == 'transpose':
            up_block = Transpose2D_block
        else:
            up_block = Upsample2D_block

        skip_connection_layers = ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2')

        # convert layer names to indices
        skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                                for l in skip_connection_layers])

        n_upsample_blocks = 5
        upsample_rates = (2,2,2,2,2)
        decoder_filters = (256,128,64,32,16)
        block_type='upsampling'
        activation='sigmoid'
        use_batchnorm=True
        classes=1

        for i in range(n_upsample_blocks):

            # check if there is a skip connection
            skip_connection = None
            if i < len(skip_connection_idx):
                skip_connection = backbone.layers[skip_connection_idx[i]].output

            upsample_rate = to_tuple(upsample_rates[i])

            x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                        skip=skip_connection, use_batchnorm=use_batchnorm)(x)

        x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
        x = Activation(activation, name=activation)(x)

        model = Model(input, x)
        res = run_image(model, self.model_files, img_path, target_size=(256, 256, 3))
        self.assertTrue(*res)

if __name__ == "__main__":
    unittest.main()
