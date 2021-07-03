# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras
from keras.initializers import RandomUniform
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_keras_and_ort, test_level_0
K = keras.backend

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
Add = keras.layers.Add
concatenate = keras.layers.concatenate
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GaussianNoise = keras.layers.GaussianNoise
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
MaxPooling2D = keras.layers.MaxPooling2D
Reshape = keras.layers.Reshape

Sequential = keras.models.Sequential
Model = keras.models.Model


def conv_layer(d, k):
    return Conv2D(d, k, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')


def conv_block(inp, d=3, pool_size=(2, 2), k=3):
    conv = conv_layer(d, k)(inp)
    return MaxPooling2D(pool_size=pool_size)(conv)


# Model from https://github.com/germain-hug/Deep-RL-Keras
class TestDeepRL(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DPPG_actor(self):
        K.clear_session()
        env_dim = (2, 3)
        act_dim = 5
        act_range = 4
        inp = Input(shape=env_dim)
        #
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        #
        out = Dense(act_dim, activation='tanh', kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i * act_range)(out)
        #
        keras_model = Model(inp, out)
        data = np.random.rand(1000, 2, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DPPG_critic(self):
        K.clear_session()
        env_dim = (2, 3)
        act_dim = (5,)
        state = Input(shape=env_dim)
        action = Input(shape=act_dim)
        x = Dense(256, activation='relu')(state)
        x = concatenate([Flatten()(x), action])
        x = Dense(128, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        keras_model = Model([state, action], out)
        data1 = np.random.rand(5, 2, 3).astype(np.float32)
        data2 = np.random.rand(5, 5).astype(np.float32)
        expected = keras_model.predict([data1, data2])
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, [data1, data2], expected, self.model_files))

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DDQN(self):
        K.clear_session()
        state_dim = (2, 3)
        action_dim = 5
        inp = Input(shape=(state_dim))

        # Determine whether we are dealing with an image input (Atari) or not
        if (len(state_dim) > 2):
            inp = Input((state_dim[1:]))
            x = conv_block(inp, 32, (2, 2), 8)
            x = conv_block(x, 64, (2, 2), 4)
            x = conv_block(x, 64, (2, 2), 3)
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
        else:
            x = Flatten()(inp)
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)

        x = Dense(action_dim + 1, activation='linear')(x)
        x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True),
                   output_shape=(action_dim,))(x)
        keras_model = Model(inp, x)
        data = np.random.rand(1000, 2, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected,
                              self.model_files, rtol=1e-2, atol=2e-2))


if __name__ == "__main__":
    unittest.main()
