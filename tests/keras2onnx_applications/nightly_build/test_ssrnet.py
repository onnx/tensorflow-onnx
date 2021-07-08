# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import onnx
import numpy as np
from mock_keras2onnx.proto import keras
from keras.applications import VGG19
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_onnx_runtime, test_level_0

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
MaxPooling2D = keras.layers.MaxPooling2D
Multiply = keras.layers.Multiply
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model
K = keras.backend

# From https://github.com/shamangary/FSA-Net
class SSR_net_MT:
    def __init__(self, image_size, num_classes, stage_num, lambda_d):

        self._channel_axis = -1
        self._input_shape = (image_size, image_size, 3)
        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

    def __call__(self):

        img_inputs = Input(self._input_shape)
        # -------------------------------------------------------------------------------------------------------------------------
        x = SeparableConv2D(16, (3, 3), padding='same')(img_inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x_layer1 = AveragePooling2D((2, 2))(x)
        x = SeparableConv2D(32, (3, 3), padding='same')(x_layer1)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x_layer2 = AveragePooling2D((2, 2))(x)
        x = SeparableConv2D(64, (3, 3), padding='same')(x_layer2)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x_layer3 = AveragePooling2D((2, 2))(x)
        x = SeparableConv2D(128, (3, 3), padding='same')(x_layer3)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x_layer4 = Activation('relu')(x)
        # -------------------------------------------------------------------------------------------------------------------------
        s = SeparableConv2D(16, (3, 3), padding='same')(img_inputs)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s_layer1 = MaxPooling2D((2, 2))(s)
        s = SeparableConv2D(32, (3, 3), padding='same')(s_layer1)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s = SeparableConv2D(32, (3, 3), padding='same')(s)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s_layer2 = MaxPooling2D((2, 2))(s)
        s = SeparableConv2D(64, (3, 3), padding='same')(s_layer2)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s = SeparableConv2D(64, (3, 3), padding='same')(s)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s_layer3 = MaxPooling2D((2, 2))(s)
        s = SeparableConv2D(128, (3, 3), padding='same')(s_layer3)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s = SeparableConv2D(128, (3, 3), padding='same')(s)
        s = BatchNormalization(axis=-1)(s)
        s_layer4 = Activation('tanh')(s)

        # -------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = Conv2D(64, (1, 1), activation='tanh')(s_layer4)
        s_layer4 = MaxPooling2D((2, 2))(s_layer4)

        x_layer4 = Conv2D(64, (1, 1), activation='relu')(x_layer4)
        x_layer4 = AveragePooling2D((2, 2))(x_layer4)

        feat_s1_pre = Multiply()([s_layer4, x_layer4])
        feat_s1_pre = Flatten()(feat_s1_pre)
        feat_delta_s1 = Dense(2 * self.num_classes, activation='tanh')(feat_s1_pre)
        delta_s1 = Dense(self.num_classes, activation='tanh', name='delta_s1')(feat_delta_s1)

        feat_local_s1 = Dense(2 * self.num_classes, activation='tanh')(feat_s1_pre)
        local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

        feat_pred_s1 = Dense(self.stage_num[0] * self.num_classes, activation='relu')(feat_s1_pre)
        pred_a_s1 = Reshape((self.num_classes, self.stage_num[0]))(feat_pred_s1)
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer3 = Conv2D(64, (1, 1), activation='tanh')(s_layer3)
        s_layer3 = MaxPooling2D((2, 2))(s_layer3)

        x_layer3 = Conv2D(64, (1, 1), activation='relu')(x_layer3)
        x_layer3 = AveragePooling2D((2, 2))(x_layer3)

        feat_s2_pre = Multiply()([s_layer3, x_layer3])
        feat_s2_pre = Flatten()(feat_s2_pre)
        feat_delta_s2 = Dense(2 * self.num_classes, activation='tanh')(feat_s2_pre)
        delta_s2 = Dense(self.num_classes, activation='tanh', name='delta_s2')(feat_delta_s2)

        feat_local_s2 = Dense(2 * self.num_classes, activation='tanh')(feat_s2_pre)
        local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

        feat_pred_s2 = Dense(self.stage_num[1] * self.num_classes, activation='relu')(feat_s2_pre)
        pred_a_s2 = Reshape((self.num_classes, self.stage_num[1]))(feat_pred_s2)
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(64, (1, 1), activation='tanh')(s_layer2)
        s_layer2 = MaxPooling2D((2, 2))(s_layer2)

        x_layer2 = Conv2D(64, (1, 1), activation='relu')(x_layer2)
        x_layer2 = AveragePooling2D((2, 2))(x_layer2)

        feat_s3_pre = Multiply()([s_layer2, x_layer2])
        feat_s3_pre = Flatten()(feat_s3_pre)
        feat_delta_s3 = Dense(2 * self.num_classes, activation='tanh')(feat_s3_pre)
        delta_s3 = Dense(self.num_classes, activation='tanh', name='delta_s3')(feat_delta_s3)

        feat_local_s3 = Dense(2 * self.num_classes, activation='tanh')(feat_s3_pre)
        local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

        feat_pred_s3 = Dense(self.stage_num[2] * self.num_classes, activation='relu')(feat_s3_pre)
        pred_a_s3 = Reshape((self.num_classes, self.stage_num[2]))(feat_pred_s3)

        # -------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x, s1, s2, s3, lambda_d):
            a = x[0][:, :, 0] * 0
            b = x[0][:, :, 0] * 0
            c = x[0][:, :, 0] * 0

            di = s1 // 2
            dj = s2 // 2
            dk = s3 // 2

            V = 99
            # lambda_d = 0.9

            for i in range(0, s1):
                a = a + (i - di + x[6]) * x[0][:, :, i]
            # a = K.expand_dims(a,-1)
            a = a / (s1 * (1 + lambda_d * x[3]))

            for j in range(0, s2):
                b = b + (j - dj + x[7]) * x[1][:, :, j]
            # b = K.expand_dims(b,-1)
            b = b / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4]))

            for k in range(0, s3):
                c = c + (k - dk + x[8]) * x[2][:, :, k]
            # c = K.expand_dims(c,-1)
            c = c / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4])) / (s3 * (1 + lambda_d * x[5]))

            pred = (a + b + c) * V

            return pred

        pred_pose = Lambda(SSR_module,
                           arguments={'s1': self.stage_num[0], 's2': self.stage_num[1], 's3': self.stage_num[2],
                                      'lambda_d': self.lambda_d}, name='pred_pose')(
            [pred_a_s1, pred_a_s2, pred_a_s3, delta_s1, delta_s2, delta_s3, local_s1, local_s2, local_s3])

        model = Model(inputs=img_inputs, outputs=pred_pose)

        return model


class TestSSRNet(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_SSR_Net_MT(self):
        K.clear_session()
        _IMAGE_SIZE = 64
        stage_num = [3, 3, 3]
        num_classes = 3
        lambda_d = 1
        keras_model = SSR_net_MT(_IMAGE_SIZE, num_classes, stage_num, lambda_d)()
        data = np.random.rand(2, 64, 64, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
