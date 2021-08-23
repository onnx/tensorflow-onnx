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
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
GlobalMaxPool1D = keras.layers.GlobalMaxPool1D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling1D = keras.layers.MaxPooling1D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
SpatialDropout1D = keras.layers.SpatialDropout1D
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


# Model from https://github.com/chen0040/keras-english-resume-parser-and-analyzer
class TestResumeParser(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_1D_CNN(self):
        K.clear_session()
        vocab_size = 50
        max_len = 20
        embedding_size = 100
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, input_length=max_len, output_dim=embedding_size))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
        model.add(GlobalMaxPool1D())
        model.add(Dense(units=30, activation='softmax'))
        data = np.random.rand(1000, max_len).astype(np.float32)
        expected = model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_Multi_Channel_CNN(self):
        K.clear_session()
        embedding_size = 100
        cnn_filter_size = 32
        length = 20
        vocab_size = 50
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, embedding_size)(inputs1)
        conv1 = Conv1D(filters=cnn_filter_size, kernel_size=4, activation='relu')(
            embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)

        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, embedding_size)(inputs2)
        conv2 = Conv1D(filters=cnn_filter_size, kernel_size=6, activation='relu')(
            embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)

        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, embedding_size)(inputs3)
        conv3 = Conv1D(filters=cnn_filter_size, kernel_size=8, activation='relu')(
            embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)

        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(10, activation='relu')(merged)

        outputs = Dense(units=30, activation='softmax')(dense1)

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        batch_size = 2
        data0 = np.random.rand(batch_size, length).astype(np.float32)
        data1 = np.random.rand(batch_size, length).astype(np.float32)
        data2 = np.random.rand(batch_size, length).astype(np.float32)
        expected = model.predict([data0, data1, data2])
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, [data0, data1, data2], expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_CNN_LSTM(self):
        K.clear_session()
        max_len = 20
        vocab_size = 50
        lstm_output_size = 70
        embedding_size = 100
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, input_length=max_len, output_dim=embedding_size))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(lstm_output_size))
        model.add(Dense(units=30, activation='softmax'))
        data = np.random.rand(2, max_len).astype(np.float32)
        expected = model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_Bidirectional_LSTM(self):
        K.clear_session()
        max_len = 20
        vocab_size = 50
        embedding_size = 100
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
        model.add(SpatialDropout1D(0.2))
        model.add(
            Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, input_shape=(max_len, embedding_size))))
        model.add(Dense(30, activation='softmax'))
        data = np.random.rand(2, max_len).astype(np.float32)
        expected = model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
