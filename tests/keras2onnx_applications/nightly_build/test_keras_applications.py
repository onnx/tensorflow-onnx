# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from mock_keras2onnx.proto import keras, is_keras_older_than
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_image, test_level_0, run_keras_and_ort
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')

Activation = keras.layers.Activation
Average = keras.layers.Average
AveragePooling2D = keras.layers.AveragePooling2D
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
Concatenate = keras.layers.Concatenate
concatenate = keras.layers.concatenate
Convolution2D = keras.layers.Convolution2D
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
MaxPool2D = keras.layers.MaxPool2D
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Model = keras.models.Model
Sequential = keras.models.Sequential

class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_MobileNet(self):
        mobilenet = keras.applications.mobilenet
        model = mobilenet.MobileNet(weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    @unittest.skipIf(is_keras_older_than("2.2.3"),
                     "There is no mobilenet_v2 module before keras 2.2.3.")
    def test_MobileNetV2(self):
        mobilenet_v2 = keras.applications.mobilenet_v2
        model = mobilenet_v2.MobileNetV2(weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_ResNet50(self):
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_InceptionV3(self):
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path, target_size=299)
        self.assertTrue(*res)

    def test_DenseNet121(self):
        from keras.applications.densenet import DenseNet121
        model = DenseNet121(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_Xception(self):
        from keras.applications.xception import Xception
        model = Xception(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=299)
        self.assertTrue(*res)

    def test_SmileCNN(self):
        # From https://github.com/kylemcdonald/SmileCNN/blob/master/2%20Training.ipynb
        nb_filters = 32
        nb_pool = 2
        nb_conv = 3
        nb_classes = 2

        model = Sequential()

        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', input_shape=(32, 32, 3)))
        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=32)
        self.assertTrue(*res)

    @unittest.skipIf(is_keras_older_than("2.2.4"),
                     "keras-resnet requires keras 2.2.4 or later.")
    def test_keras_resnet_batchnormalization(self):
        N, C, H, W = 2, 3, 120, 120
        import keras_resnet

        model = Sequential()
        model.add(ZeroPadding2D(padding=((3, 3), (3, 3)), input_shape=(H, W, C), data_format='channels_last'))
        model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=False,
                         data_format='channels_last'))
        model.add(keras_resnet.layers.BatchNormalization(freeze=True, axis=3))

        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(N, H, W, C).astype(np.float32).reshape((N, H, W, C))
        expected = model.predict(data)
        self.assertTrue(run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    # model from https://github.com/titu1994/Image-Super-Resolution
    def test_ExpantionSuperResolution(self):
        init = Input(shape=(32, 32, 3))
        x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='level1')(init)
        x1 = Convolution2D(32, (1, 1), activation='relu', padding='same', name='lavel1_1')(x)
        x2 = Convolution2D(32, (3, 3), activation='relu', padding='same', name='lavel1_2')(x)
        x3 = Convolution2D(32, (5, 5), activation='relu', padding='same', name='lavel1_3')(x)
        x = Average()([x1, x2, x3])
        out = Convolution2D(3, (5, 5), activation='relu', padding='same', name='output')(x)
        model = keras.models.Model(init, out)
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=32)
        self.assertTrue(*res)

    def test_tcn(self):
        from tcn import TCN
        batch_size, timesteps, input_dim = None, 20, 1
        actual_batch_size = 3
        i = Input(batch_shape=(batch_size, timesteps, input_dim))
        np.random.seed(1000)  # set the random seed to avoid the output result discrepancies.
        for return_sequences in [True, False]:
            o = TCN(return_sequences=return_sequences)(i)  # The TCN layers are here.
            o = Dense(1)(o)
            model = keras.models.Model(inputs=[i], outputs=[o])
            onnx_model = mock_keras2onnx.convert_keras(model, model.name)
            data = np.random.rand(actual_batch_size, timesteps, input_dim).astype(np.float32).reshape((actual_batch_size, timesteps, input_dim))
            expected = model.predict(data)
            self.assertTrue(run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    # model from https://github.com/titu1994/LSTM-FCN
    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_lstm_fcn(self):
        MAX_SEQUENCE_LENGTH = 176
        NUM_CELLS = 8
        NB_CLASS = 37
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

        x = LSTM(NUM_CELLS)(ip)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(NB_CLASS, activation='softmax')(x)

        model = Model(ip, out)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        batch_size = 2
        data = np.random.rand(batch_size, 1, MAX_SEQUENCE_LENGTH).astype(np.float32).reshape(batch_size, 1, MAX_SEQUENCE_LENGTH)
        expected = model.predict(data)
        self.assertTrue(run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    # model from https://github.com/CyberZHG/keras-self-attention
    @unittest.skipIf(test_level_0 or get_maximum_opset_supported() < 11,
                     "Test level 0 only.")
    def test_keras_self_attention(self):
        from keras_self_attention import SeqSelfAttention
        keras.backend.clear_session()

        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=10000,
                                         output_dim=300,
                                         mask_zero=True))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                               return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(keras.layers.Dense(units=5))
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(5, 10).astype(np.float32).reshape(5, 10)
        expected = model.predict(data)
        self.assertTrue(run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    # Model from https://github.com/chandrikadeb7/Face-Mask-Detection
    @unittest.skipIf(test_level_0 or is_keras_older_than("2.2.3"),
                     "There is no mobilenet_v2 module before keras 2.2.3.")
    def test_FaceMaskDetection(self):
        mobilenet_v2 = keras.applications.mobilenet_v2
        baseModel = mobilenet_v2.MobileNetV2(weights=None, include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    # Model from https://github.com/abhishekrana/DeepFashion
    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DeepFashion(self):
        base_model = keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        model_inputs = base_model.input
        common_inputs = base_model.output
        dropout_rate = 0.5
        output_classes = 20
        x = Flatten()(common_inputs)
        x = Dense(256, activation='tanh')(x)
        x = Dropout(dropout_rate)(x)
        predictions_class = Dense(output_classes, activation='softmax', name='predictions_class')(x)

        ## Model (Regression) IOU score
        x = Flatten()(common_inputs)
        x = Dense(256, activation='tanh')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation='tanh')(x)
        x = Dropout(dropout_rate)(x)
        predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)

        ## Create Model
        keras_model = Model(inputs=model_inputs, outputs=[predictions_class, predictions_iou])
        res = run_image(keras_model, self.model_files, img_path, atol=5e-3, target_size=224)
        self.assertTrue(*res)

    # Model from https://github.com/manicman1999/Keras-BiGAN
    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_bigan_generator(self):
        def g_block(inp, fil, u=True):

            if u:
                out = UpSampling2D(interpolation='bilinear')(inp)
            else:
                out = Activation('linear')(inp)

            skip = Conv2D(fil, 1, padding='same', kernel_initializer='he_normal')(out)

            out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
            out = LeakyReLU(0.2)(out)

            out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
            out = LeakyReLU(0.2)(out)

            out = Conv2D(fil, 1, padding='same', kernel_initializer='he_normal')(out)

            out = keras.layers.add([out, skip])
            out = LeakyReLU(0.2)(out)

            return out

        latent_size = 64
        cha = 16

        inp = Input(shape = [latent_size])

        x = Dense(4*4*16*cha, kernel_initializer = 'he_normal')(inp)
        x = Reshape([4, 4, 16*cha])(x)

        x = g_block(x, 16 * cha, u = False)  #4
        x = g_block(x, 8 * cha)  #8
        x = g_block(x, 4 * cha)  #16
        x = g_block(x, 3 * cha)   #32
        x = g_block(x, 2 * cha)   #64
        x = g_block(x, 1 * cha)   #128

        x = Conv2D(filters = 3, kernel_size = 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(x)

        model = Model(inputs = inp, outputs = x)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(200, latent_size).astype(np.float32).reshape(200, latent_size)
        expected = model.predict(data)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    # Model from https://github.com/ankur219/ECG-Arrhythmia-classification
    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_ecg_classification(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=[128, 128, 3], kernel_initializer='glorot_uniform'))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(2048))
        model.add(keras.layers.ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(2, 128, 128, 3).astype(np.float32)
        expected = model.predict(data)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model, data, expected, self.model_files))

    # Model from https://github.com/arunponnusamy/gender-detection-keras
    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_gender_detection(self):
        model = Sequential()
        inputShape = (224, 224, 3)
        chanDim = -1
        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(80))
        model.add(Activation("sigmoid"))

        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=224)
        self.assertTrue(*res)

if __name__ == "__main__":
    unittest.main()
