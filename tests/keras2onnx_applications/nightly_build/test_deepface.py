# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_onnx_runtime, test_level_0

Activation = keras.layers.Activation
add = keras.layers.add
Add = keras.layers.Add
concatenate = keras.layers.concatenate
Conv2D = keras.layers.Conv2D
Convolution2D = keras.layers.Convolution2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
LocallyConnected2D = keras.layers.LocallyConnected2D
MaxPooling2D = keras.layers.MaxPooling2D
Multiply = keras.layers.Multiply
Reshape = keras.layers.Reshape

Sequential = keras.models.Sequential
Model = keras.models.Model


# model from https://github.com/serengil/deepface
class TestDeepFace(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skip("LocallyConnected2D conversion is slow.")
    def test_DeepFace(self):
        base_model = Sequential()
        base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
        base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
        base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
        base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
        base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5'))
        base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
        base_model.add(Flatten(name='F0'))
        base_model.add(Dense(4096, activation='relu', name='F7'))
        base_model.add(Dropout(rate=0.5, name='D0'))
        base_model.add(Dense(8631, activation='softmax', name='F8'))
        data = np.random.rand(1, 152, 152, 3).astype(np.float32)
        expected = base_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(base_model, base_model.name, debug_mode=True)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DeepID(self):
        myInput = Input(shape=(55, 47, 3))

        x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=(55, 47, 3))(myInput)
        x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
        x = Dropout(rate=1, name='D1')(x)

        x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
        x = Dropout(rate=1, name='D2')(x)

        x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
        x = Dropout(rate=1, name='D3')(x)

        x1 = Flatten()(x)
        fc11 = Dense(160, name='fc11')(x1)

        x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
        x2 = Flatten()(x2)
        fc12 = Dense(160, name='fc12')(x2)

        y = Add()([fc11, fc12])
        y = Activation('relu', name='deepid')(y)

        keras_model = Model(inputs=[myInput], outputs=y)
        data = np.random.rand(50, 55, 47, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
