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

Activation = keras.layers.Activation
add = keras.layers.add
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Conv2D = keras.layers.Conv2D
Convolution2D = keras.layers.Convolution2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
GRU = keras.layers.GRU
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling2D = keras.layers.MaxPooling2D
Multiply = keras.layers.Multiply
Reshape = keras.layers.Reshape

Sequential = keras.models.Sequential
Model = keras.models.Model
K = keras.backend


# Model from https://github.com/qjadud1994/CRNN-Keras
class TestCRNN(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(get_maximum_opset_supported() < 10,
                     "CRNN conversion need opset >= 10.")
    def test_CRNN_LSTM(self):
        img_w = 128
        img_h = 64
        input_shape = (img_w, img_h, 1)  # (128, 64, 1)
        num_classes = 80

        # Make Networkw
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

        # Convolution layer (VGG)
        inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
            inputs)  # (None, 128, 64, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

        inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
            inner)  # (None, 64, 32, 128)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

        inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

        inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
            inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

        inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
            inner)  # (None, 32, 4, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)

        # CNN to RNN
        inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

        # RNN layer
        lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(
            inner)  # (None, 32, 512)
        lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
            inner)
        reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

        lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
        lstm1_merged = BatchNormalization()(lstm1_merged)

        lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
            lstm1_merged)
        reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)

        lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 1024)
        lstm2_merged = BatchNormalization()(lstm2_merged)

        # transforms RNN output to character activations:
        inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, 32, 63)
        y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=[inputs], outputs=y_pred)

        data = np.random.rand(1, 128, 64, 1).astype(np.float32)
        expected = model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    @unittest.skipIf(get_maximum_opset_supported() < 10,
                     "CRNN conversion need opset >= 10.")
    def test_CRNN_GRU(self):
        img_w = 128
        img_h = 64
        num_classes = 80
        input_shape = (img_w, img_h, 1)  # (128, 64, 1)

        # Make Networkw
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

        # Convolution layer (VGG)
        inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
            inputs)  # (None, 128, 64, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

        inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
            inner)  # (None, 64, 32, 128)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

        inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

        inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
            inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

        inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
            inner)  # (None, 32, 4, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)

        # CNN to RNN
        inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

        # RNN layer
        gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 32, 512)
        gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)

        gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
        gru1_merged = BatchNormalization()(gru1_merged)

        gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)
        reversed_gru_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_2b)

        gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
        gru2_merged = BatchNormalization()(gru2_merged)

        # transforms RNN output to character activations:
        inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(gru2_merged)  # (None, 32, 63)
        y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=[inputs], outputs=y_pred)

        data = np.random.rand(1, 128, 64, 1).astype(np.float32)
        expected = model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
