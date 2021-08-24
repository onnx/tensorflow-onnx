# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import keras_contrib
import numpy as np
from mock_keras2onnx import set_converter
from mock_keras2onnx.proto import keras
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_onnx_runtime, convert_InstanceNormalizationLayer

Activation = keras.layers.Activation
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
InstanceNormalization = keras_contrib.layers.InstanceNormalization
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model

# From https://github.com/eriklindernoren/Keras-GAN/blob/master/pixelda/pixelda.py
class PixelDA():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10

        # Loss weights
        lambda_adv = 10
        lambda_clf = 1

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of residual blocks in the generator
        self.residual_blocks = 6

        # Number of filters in first layer of discriminator and classifier
        self.df = 64
        self.cf = 64

        # Build and compile the discriminators
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # Build the task (classification) network
        self.clf = self.build_classifier()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images from domain A to domain B
        fake_B = self.generator(img_A)

        # Classify the translated image
        class_pred = self.clf(fake_B)

        # For the combined model we will only train the generator and classifier
        self.discriminator.trainable = False

        # Discriminator determines validity of translated images
        valid = self.discriminator(fake_B)

        self.combined = Model(img_A, [valid, class_pred])

    def build_generator(self):
        """Resnet Generator"""

        def residual_block(layer_input):
            """Residual block described in paper"""
            d = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = Activation('relu')(d)
            d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        # Image input
        img = Input(shape=self.img_shape)

        l1 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(img)

        # Propogate signal through residual blocks
        r = residual_block(l1)
        for _ in range(self.residual_blocks - 1):
            r = residual_block(r)

        output_img = Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')(r)

        return Model(img, output_img)


    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def build_classifier(self):

        def clf_layer(layer_input, filters, f_size=4, normalization=True):
            """Classifier layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        c1 = clf_layer(img, self.cf, normalization=False)
        c2 = clf_layer(c1, self.cf*2)
        c3 = clf_layer(c2, self.cf*4)
        c4 = clf_layer(c3, self.cf*8)
        c5 = clf_layer(c4, self.cf*8)

        class_pred = Dense(self.num_classes, activation='softmax')(Flatten()(c5))

        return Model(img, class_pred)


set_converter(keras_contrib.layers.InstanceNormalization, convert_InstanceNormalizationLayer)


class TestPixelDA(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_PixelDA(self):
        keras_model = PixelDA().combined
        x = np.random.rand(5, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict([x])
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files, atol=1.e-5))


if __name__ == "__main__":
    unittest.main()
