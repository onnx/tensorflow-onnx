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
from keras import layers
from keras import regularizers

NUM_FRAMES = 160
NUM_FBANKS = 64

class DeepSpeakerModel:

    def __init__(self, batch_input_shape=(None, NUM_FRAMES, NUM_FBANKS, 1), include_softmax=False,
                 num_speakers_softmax=None):
        self.include_softmax = include_softmax
        if self.include_softmax:
            assert num_speakers_softmax > 0
        self.clipped_relu_count = 0

        inputs = Input(batch_shape=batch_input_shape, name='input')
        x = self.cnn_component(inputs)

        x = Reshape((-1, 2048))(x)
        # Temporal average layer. axis=1 is time.
        x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
        if include_softmax:
            # used for softmax because the dataset we pre-train on might be too small. easy to overfit.
            x = Dropout(0.5)(x)
        x = Dense(512, name='affine')(x)
        if include_softmax:
            # Those weights are just when we train on softmax.
            x = Dense(num_speakers_softmax, activation='softmax')(x)
        else:
            # Does not contain any weights.
            x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
        self.m = Model(inputs, x, name='ResCNN')

    def keras_model(self):
        return self.m

    def get_weights(self):
        w = self.m.get_weights()
        if self.include_softmax:
            w.pop()  # last 2 are the W_softmax and b_softmax.
            w.pop()
        return w

    def clipped_relu(self, inputs):
        relu = Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name='clipped_relu_'+str(self.clipped_relu_count))(inputs)
        self.clipped_relu_count += 1
        return relu

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'

        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2a')(input_tensor)
        x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
        x = self.clipped_relu(x)

        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2b')(x)
        x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)

        x = self.clipped_relu(x)

        x = layers.add([x, input_tensor])
        x = self.clipped_relu(x)
        return x

    def conv_and_res_block(self, inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                   kernel_size=5,
                   strides=2,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = self.clipped_relu(o)
        for i in range(3):
            o = self.identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(self, inp):
        x = self.conv_and_res_block(inp, 64, stage=1)
        x = self.conv_and_res_block(x, 128, stage=2)
        x = self.conv_and_res_block(x, 256, stage=3)
        x = self.conv_and_res_block(x, 512, stage=4)
        return x


# Model from https://github.com/philipperemy/deep-speaker
class TestDeepSpeaker(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DeepSpeaker(self):
        K.clear_session()
        keras_model = DeepSpeakerModel(batch_input_shape=(None, 32, 64, 4), include_softmax=True, num_speakers_softmax=10).keras_model()
        data = np.random.rand(2, 32, 64, 4).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
