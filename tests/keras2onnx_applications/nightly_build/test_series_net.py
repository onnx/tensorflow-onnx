# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_keras_and_ort, test_level_0
K = keras.backend

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Conv1D = keras.layers.Conv1D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
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


def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        residual = input_

        layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length,
                           dilation_rate=dilation,
                           activation='linear', padding='causal', use_bias=False,
                           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                              seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)

        layer_out = Activation('selu')(layer_out)

        skip_out = Conv1D(1, 1, activation='linear', use_bias=False,
                          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                             seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_in = Conv1D(1, 1, activation='linear', use_bias=False,
                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                               seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_out = Add()([residual, network_in])

        return network_out, skip_out

    return f


def DC_CNN_Model(length):
    input = Input(shape=(length, 1))

    l1a, l1b = DC_CNN_Block(32, 2, 1, 0.001)(input)
    l2a, l2b = DC_CNN_Block(32, 2, 2, 0.001)(l1a)
    l3a, l3b = DC_CNN_Block(32, 2, 4, 0.001)(l2a)
    l4a, l4b = DC_CNN_Block(32, 2, 8, 0.001)(l3a)
    l5a, l5b = DC_CNN_Block(32, 2, 16, 0.001)(l4a)
    l6a, l6b = DC_CNN_Block(32, 2, 32, 0.001)(l5a)
    l6b = Dropout(0.8)(l6b)  # dropout used to limit influence of earlier data
    l7a, l7b = DC_CNN_Block(32, 2, 64, 0.001)(l6a)
    l7b = Dropout(0.8)(l7b)  # dropout used to limit influence of earlier data

    l8 = Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])

    l9 = Activation('relu')(l8)

    l21 = Conv1D(1, 1, activation='linear', use_bias=False,
                 kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                 kernel_regularizer=l2(0.001))(l9)

    model = Model(input=input, output=l21)

    return model


# Model from https://github.com/kristpapadopoulos/seriesnet
class TestSeriesNet(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(False,
                     "Test level 0 only.")
    def test_series_net(self):
        K.clear_session()
        keras_model = DC_CNN_Model(20)
        data = np.random.rand(2000, 20, 1).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
