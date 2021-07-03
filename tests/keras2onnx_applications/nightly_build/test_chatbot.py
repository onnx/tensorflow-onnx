# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from mock_keras2onnx.proto.tfcompat import is_tf2
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
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling2D = keras.layers.MaxPooling2D
Multiply = keras.layers.Multiply
Permute = keras.layers.Permute
RepeatVector = keras.layers.RepeatVector
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


# Model from https://github.com/Dimsmary/Ossas_ChatBot
@unittest.skipIf(not is_tf2, "Tensorflow 2.x only tests")
class TestChatBot(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_chatbot(self):
        K.clear_session()
        vocabulary_size = 1085
        embedding_dim = int(pow(vocabulary_size, 1.0 / 4))
        latent_dim = embedding_dim * 40
        encoder_inputs = Input(shape=(None,), name='encoder_input')
        encoder_embedding = Embedding(vocabulary_size,
                                      embedding_dim,
                                      mask_zero=True,
                                      name='encoder_Embedding')(encoder_inputs)
        encoder = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.5),
                                name='encoder_BiLSTM')
        encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = encoder(encoder_embedding)
        state_h = Concatenate(axis=-1, name='encoder_state_h')([fw_state_h, bw_state_h])
        state_c = Concatenate(axis=-1, name='encoder_state_c')([fw_state_c, bw_state_c])
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(vocabulary_size,
                                      embedding_dim,
                                      mask_zero=True,
                                      name='decoder_embedding')(decoder_inputs)
        decoder_lstm = LSTM(latent_dim * 2,
                            return_sequences=True,
                            return_state=True,
                            name='decoder_LSTM',
                            dropout=0.5)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding,
                                             initial_state=encoder_states)

        attention = Dense(1, activation='tanh')(encoder_outputs)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(latent_dim * 2)(attention)
        attention = Permute([2, 1])(attention)
        sent_dense = Multiply()([decoder_outputs, attention])

        decoder_dense = Dense(vocabulary_size, activation='softmax', name='dense_layer')
        decoder_outputs = decoder_dense(sent_dense)
        keras_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        data1 = np.random.rand(2, 12).astype(np.float32)
        data2 = np.random.rand(2, 12).astype(np.float32)
        expected = keras_model.predict([data1, data2])
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, [data1, data2], expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
