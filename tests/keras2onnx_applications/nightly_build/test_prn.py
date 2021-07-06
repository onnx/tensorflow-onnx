# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_keras_and_ort, test_level_0
K = keras.backend

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Conv2D = keras.layers.Conv2D
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


def PRN(height, width, node_count):
    input = Input(shape=(height, width, 17))
    y = Flatten()(input)
    x = Dense(node_count, activation='relu')(y)
    x = Dropout(0.5)(x)
    x = Dense(width * height * 17, activation='relu')(x)
    x = keras.layers.Add()([x, y])
    x = keras.layers.Activation('softmax')(x)
    x = Reshape((height, width, 17))(x)
    model = Model(inputs=input, outputs=x)
    return model


def PRN_Seperate(height, width, node_count):
    input = Input(shape=(height, width, 17))
    y = Flatten()(input)
    x = Dense(node_count, activation='relu')(y)
    x = Dropout(0.5)(x)
    x = Dense(width * height * 17, activation='relu')(x)
    x = keras.layers.Add()([x, y])
    out = []
    start = 0
    end = width * height

    for i in range(17):
        o = keras.layers.Lambda(lambda x: x[:, start:end])(x)
        o = Activation('softmax')(o)
        out.append(o)
        start = end
        end = start + width * height

    x = keras.layers.Concatenate()(out)
    x = Reshape((height, width, 17))(x)
    model = Model(inputs=input, outputs=x)
    return model


# Model from https://github.com/mkocabas/pose-residual-network
class TestPRN(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_PRN(self):
        K.clear_session()
        keras_model = PRN(28, 18, 15)
        data = np.random.rand(2, 28, 18, 17).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_PRN_Separate(self):
        K.clear_session()
        keras_model = PRN_Seperate(28, 18, 15)
        data = np.random.rand(2, 28, 18, 17).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
