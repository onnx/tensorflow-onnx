# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import keras
import keras_segmentation
import numpy as np
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_image
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling1D = keras.layers.MaxPooling1D
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D

class TestPSPNet(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def _pool_block(self, feats, pool_factor, IMAGE_ORDERING):
        import keras.backend as K
        if IMAGE_ORDERING == 'channels_first':
            h = K.int_shape(feats)[2]
            w = K.int_shape(feats)[3]
        elif IMAGE_ORDERING == 'channels_last':
            h = K.int_shape(feats)[1]
            w = K.int_shape(feats)[2]
        pool_size = strides = [int(np.round(float(h) / pool_factor)), int(np.round(float(w) / pool_factor))]
        x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING, strides=strides, padding='same')(feats)
        x = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = keras_segmentation.models.model_utils.resize_image(x, strides, data_format=IMAGE_ORDERING)
        return x

    def test_pspnet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/pspnet.py
        from keras_segmentation.models.basic_models import vanilla_encoder
        img_input, levels = vanilla_encoder(input_height=384, input_width=576)
        o = levels[4]
        pool_factors = [1, 2, 3, 6]
        pool_outs = [o]
        IMAGE_ORDERING = 'channels_last'
        if IMAGE_ORDERING == 'channels_first':
            MERGE_AXIS = 1
        elif IMAGE_ORDERING == 'channels_last':
            MERGE_AXIS = -1
        for p in pool_factors:
            pooled = self._pool_block(o, p, IMAGE_ORDERING)
            pool_outs.append(pooled)
        o = Concatenate(axis=MERGE_AXIS)(pool_outs)
        o = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, use_bias=False)(o)
        o = BatchNormalization()(o)
        o = Activation('relu')(o)
        o = Conv2D(101, (3, 3), data_format=IMAGE_ORDERING, padding='same')(o)
        o = keras_segmentation.models.model_utils.resize_image(o, (8, 8), data_format=IMAGE_ORDERING)

        model = keras_segmentation.models.model_utils.get_segmentation_model(img_input, o)
        model.model_name = "pspnet"

        res = run_image(model, self.model_files, img_path, target_size=(384, 576))
        self.assertTrue(*res)


if __name__ == "__main__":
    unittest.main()
