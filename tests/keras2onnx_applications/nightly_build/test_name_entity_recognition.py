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
Bidirectional = keras.layers.Bidirectional
concatenate = keras.layers.concatenate
Conv1D = keras.layers.Conv1D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GaussianNoise = keras.layers.GaussianNoise
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LSTM = keras.layers.LSTM
MaxPooling1D = keras.layers.MaxPooling1D
Reshape = keras.layers.Reshape
TimeDistributed = keras.layers.TimeDistributed

Sequential = keras.models.Sequential
Model = keras.models.Model


# Model from https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs
class TestNameEntityRecognition(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(get_maximum_opset_supported() < 11,
                     "Deep speech conversion need opset >= 11.")
    def test_name_entity_recognition(self):
        K.clear_session()
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=10, output_dim=20,
                          weights=None, trainable=False)(words_input)
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=20, input_dim=12,
                           weights=None, trainable=False)(casing_input)
        character_input = Input(shape=(None, 52,), name='char_input')
        embed_char_out = TimeDistributed(
            Embedding(26, 20),
            name='char_embedding')(character_input)
        dropout = Dropout(0.5)(embed_char_out)
        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(
            dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
        char = TimeDistributed(Flatten())(maxpool_out)
        char = Dropout(0.5)(char)
        output = concatenate([words, casing, char])
        output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
        output = TimeDistributed(Dense(35, activation='softmax'))(output)
        keras_model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        batch_size = 100
        data1 = np.random.randint(5, 10, size=(batch_size, 6)).astype(np.int32)
        data2 = np.random.randint(5, 10, size=(batch_size, 6)).astype(np.int32)
        data3 = np.random.rand(batch_size, 6, 52).astype(np.float32)
        expected = keras_model.predict([data1, data2, data3])
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model,
                              {keras_model.input_names[0]: data1,
                               keras_model.input_names[1]: data2,
                               keras_model.input_names[2]: data3}, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
