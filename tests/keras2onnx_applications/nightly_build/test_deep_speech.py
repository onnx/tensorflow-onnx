# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
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
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model
layers = keras.layers


# Model from https://github.com/rolczynski/Automatic-Speech-Recognition
class TestDeepSpeech(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(get_maximum_opset_supported() < 11,
                     "Deep speech conversion need opset >= 11.")
    def test_deep_speech(self):
        K.clear_session()
        input_dim = 20
        output_dim = 10
        context = 7
        units = 1024
        dropouts = (0.1, 0.1, 0)

        # Define input tensor [batch, time, features]
        input_tensor = layers.Input([None, input_dim], name='X')

        # Add 4th dimension [batch, time, frequency, channel]
        x = layers.Lambda(keras.backend.expand_dims,
                          arguments=dict(axis=-1))(input_tensor)
        # Fill zeros around time dimension
        x = layers.ZeroPadding2D(padding=(context, 0))(x)
        # Convolve signal in time dim
        receptive_field = (2 * context + 1, input_dim)
        x = layers.Conv2D(filters=units, kernel_size=receptive_field)(x)
        # Squeeze into 3rd dim array
        x = layers.Lambda(keras.backend.squeeze, arguments=dict(axis=2))(x)
        # Add non-linearity
        x = layers.ReLU(max_value=20)(x)
        # Use dropout as regularization
        x = layers.Dropout(rate=dropouts[0])(x)

        # 2nd and 3rd FC layers do a feature extraction base on a narrow
        # context of convolutional layer
        x = layers.TimeDistributed(layers.Dense(units))(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[1])(x)

        x = layers.TimeDistributed(layers.Dense(units))(x)
        x = layers.ReLU(max_value=20)(x)
        x = layers.Dropout(rate=dropouts[2])(x)

        # Use recurrent layer to have a broader context
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True),
                                 merge_mode='sum')(x)

        # Return at each time step logits along characters. Then CTC
        # computation is more stable, in contrast to the softmax.
        output_tensor = layers.TimeDistributed(layers.Dense(output_dim))(x)
        model = keras.Model(input_tensor, output_tensor, name='DeepSpeech')
        data = np.random.rand(2, 3, input_dim).astype(np.float32)
        expected = model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    @unittest.skipIf(get_maximum_opset_supported() < 11,
                     "Deep speech conversion need opset >= 11.")
    def test_deep_speech_2(self):
        K.clear_session()
        input_dim = 20
        output_dim = 10
        rnn_units = 800
        # Define input tensor [batch, time, features]
        input_tensor = layers.Input([None, input_dim], name='X')

        # Add 4th dimension [batch, time, frequency, channel]
        x = layers.Lambda(keras.backend.expand_dims,
                          arguments=dict(axis=-1))(input_tensor)
        x = layers.Conv2D(filters=32,
                          kernel_size=[11, 41],
                          strides=[2, 2],
                          padding='same',
                          use_bias=False,
                          name='conv_1')(x)
        x = layers.BatchNormalization(name='conv_1_bn')(x)
        x = layers.ReLU(name='conv_1_relu')(x)

        x = layers.Conv2D(filters=32,
                          kernel_size=[11, 21],
                          strides=[1, 2],
                          padding='same',
                          use_bias=False,
                          name='conv_2')(x)
        x = layers.BatchNormalization(name='conv_2_bn')(x)
        x = layers.ReLU(name='conv_2_relu')(x)
        # We need to squeeze to 3D tensor. Thanks to the stride in frequency
        # domain, we reduce the number of features four times for each channel.
        x = layers.Reshape([-1, input_dim//4*32])(x)

        for i in [1, 2, 3, 4, 5]:
            recurrent = layers.GRU(units=rnn_units,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   use_bias=True,
                                   return_sequences=True,
                                   reset_after=True,
                                   name='gru_'+str(i))
            x = layers.Bidirectional(recurrent,
                                     name='bidirectional'+str(i),
                                     merge_mode='concat')(x)
            x = layers.Dropout(rate=0.5)(x) if i < 5 else x  # Only between

        # Return at each time step logits along characters. Then CTC
        # computation is more stable, in contrast to the softmax.
        x = layers.TimeDistributed(layers.Dense(units=rnn_units*2), name='dense_1')(x)
        x = layers.ReLU(name='dense_1_relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        output_tensor = layers.TimeDistributed(layers.Dense(units=output_dim),
                                               name='dense_2')(x)

        model = keras.Model(input_tensor, output_tensor, name='DeepSpeech2')
        data = np.random.rand(2, 3, input_dim).astype(np.float32)
        expected = model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
