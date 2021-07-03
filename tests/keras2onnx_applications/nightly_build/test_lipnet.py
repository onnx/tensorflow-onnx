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
Bidirectional = keras.layers.Bidirectional
concatenate = keras.layers.concatenate
Conv3D = keras.layers.Conv3D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
GRU = keras.layers.GRU
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
MaxPooling3D = keras.layers.MaxPooling3D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
TimeDistributed = keras.layers.TimeDistributed
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D
ZeroPadding3D = keras.layers.ZeroPadding3D

Sequential = keras.models.Sequential
Model = keras.models.Model


# Model from https://github.com/rizkiarm/LipNet
class TestLipNet(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_LipNet(self):
        K.clear_session()
        frames_n = 2
        img_w = 128
        img_h = 128
        img_c = 3
        output_size = 28
        input_shape = (frames_n, img_w, img_h, img_c)
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_data)
        conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), activation='relu', kernel_initializer='he_normal', name='conv1')(zero1)
        maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(conv1)
        drop1 = Dropout(0.5)(maxp1)

        zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(drop1)
        conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv2')(zero2)
        maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(conv2)
        drop2 = Dropout(0.5)(maxp2)

        zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(drop2)
        conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv3')(zero3)
        maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(conv3)
        drop3 = Dropout(0.5)(maxp3)

        resh1 = TimeDistributed(Flatten())(drop3)

        gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(resh1)
        gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(gru_1)

        # transforms RNN output to character activations:
        dense1 = Dense(output_size, kernel_initializer='he_normal', name='dense1')(gru_2)

        y_pred = Activation('softmax', name='softmax')(dense1)

        keras_model = Model(inputs=input_data, outputs=y_pred)
        data = np.random.rand(2, frames_n, img_w, img_h, img_c).astype(np.float32)
        expected = keras_model.predict([data])
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
