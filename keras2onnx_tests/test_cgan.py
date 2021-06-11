# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
import keras2onnx
import numpy as np
from keras2onnx.proto import keras, is_tf_keras
from distutils.version import StrictVersion

Activation = keras.layers.Activation
BatchNormalization = keras.layers.BatchNormalization
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D

Sequential = keras.models.Sequential
Model = keras.models.Model


# From https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)

    def get_model(self):
        return self.combined

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)


@pytest.mark.skipif(keras2onnx.proto.tfcompat.is_tf2 and is_tf_keras, reason="Tensorflow 1.x only tests.")
@pytest.mark.skipif(is_tf_keras and StrictVersion(tf.__version__.split('-')[0]) < StrictVersion("1.14.0"),
                    reason="Not supported before tensorflow 1.14.0 for tf_keras")
def test_CGAN(runner):
    keras_model = CGAN().combined
    batch = 5
    x = np.random.rand(batch, 100).astype(np.float32)
    y = np.random.rand(batch, 1).astype(np.float32)
    expected = keras_model.predict([x, y])
    onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
    assert runner(onnx_model.graph.name, onnx_model,
                  {keras_model.input_names[0]: x, keras_model.input_names[1]: y}, expected)
