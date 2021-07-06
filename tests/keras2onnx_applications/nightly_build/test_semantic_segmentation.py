# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
import math
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from mock_keras2onnx.proto import keras
from keras.initializers import RandomNormal
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_keras_and_ort, test_level_0
K = keras.backend
import tensorflow as tf

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
add = keras.layers.add
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
DepthwiseConv2D = keras.layers.DepthwiseConv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
MaxPooling2D = keras.layers.MaxPooling2D
Multiply = keras.layers.Multiply
multiply = keras.layers.multiply
Permute = keras.layers.Permute
PReLU = keras.layers.PReLU
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
SpatialDropout2D = keras.layers.SpatialDropout2D
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


def ConvBNLayer(x, out_channels, kernel_size, stride=1, dilation=1, act=True):
    x = Conv2D(out_channels, kernel_size, strides=stride,
               padding='same', dilation_rate=dilation)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    if act:
        return Activation('relu')(x)
    else:
        return x


def ACBlock(x, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, deploy=False):
    if deploy:
        return Conv2D(out_channels, (kernel_size, kernel_size), strides=stride,
                      dilation_rate=dilation, use_bias=True, padding='same')(x)
    else:
        square_outputs = Conv2D(out_channels, (kernel_size, kernel_size), strides=stride,
                                dilation_rate=dilation, use_bias=False, padding='same')(x)
        square_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(square_outputs)

        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (padding, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, padding)

        if center_offset_from_origin_border >= 0:
            vertical_outputs = x
            ver_conv_padding = ver_pad_or_crop
            horizontal_outputs = x
            hor_conv_padding = hor_pad_or_crop
        else:
            vertical_outputs = x

            ver_conv_padding = (0, 0)
            horizontal_outputs = x

            hor_conv_padding = (0, 0)

        vertical_outputs = ZeroPadding2D(padding=ver_conv_padding)(vertical_outputs)
        vertical_outputs = Conv2D(out_channels, kernel_size=(kernel_size, 1),
                                  strides=stride, padding='same', use_bias=False,
                                  dilation_rate=dilation)(vertical_outputs)
        vertical_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(vertical_outputs)

        horizontal_outputs = ZeroPadding2D(padding=hor_conv_padding)(horizontal_outputs)
        horizontal_outputs = Conv2D(out_channels, kernel_size=(kernel_size, 1),
                                    strides=stride, padding='same', use_bias=False,
                                    dilation_rate=dilation)(horizontal_outputs)
        horizontal_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(horizontal_outputs)

        results = Add()([square_outputs, vertical_outputs, horizontal_outputs])

        return results


def BasicBlock(x, out_channels, stride=1, downsample=False):
    residual = x
    x = ACBlock(x, out_channels, kernel_size=3, stride=stride)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)
    x = ACBlock(x, out_channels, kernel_size=3)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)

    if downsample:
        shortcut = ConvBNLayer(residual, out_channels, kernel_size=1, stride=stride)
        outputs = Add()([x, shortcut])
    else:
        outputs = Add()([x, residual])

    return Activation('relu')(outputs)


def BottleNeckBlock(x, out_channels, stride=1, downsample=False):
    expansion = 4

    residual = x

    x = ConvBNLayer(x, out_channels, kernel_size=1, act=True)

    x = ACBlock(x, out_channels, kernel_size=3, stride=stride)

    x = ConvBNLayer(x, out_channels * expansion, kernel_size=1, act=False)

    if downsample:
        shortcut_tensor = ConvBNLayer(residual, out_channels * 4, kernel_size=1, stride=stride)
    else:
        shortcut_tensor = residual

    outputs = Add()([x, shortcut_tensor])
    return Activation('relu')(outputs)


def ResNet(x, block_type, layers_repeat, class_dim=1000):
    num_filters = [64, 128, 256, 512]

    x = ConvBNLayer(x, 64, kernel_size=7, stride=2, act=True)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for block in range(4):
        downsample = True
        for i in range(layers_repeat[block]):
            x = block_type(x, num_filters[block], stride=2 if i == 0 and block != 0 else 1, downsample=downsample)
            downsample = False

    pool = GlobalAveragePooling2D()(x)
    output = Dense(class_dim, activation='relu')(pool)
    return output


def ResACNet(x, class_dim=1000, depth=50):
    assert depth in [10, 18, 34, 50, 101, 152, 200]

    if depth == 10:
        output = ResNet(x, BasicBlock, [1, 1, 1, 1], class_dim)
    elif depth == 18:
        output = ResNet(x, BasicBlock, [2, 2, 2, 2], class_dim)
    elif depth == 34:
        output = ResNet(x, BasicBlock, [3, 4, 6, 3], class_dim)
    elif depth == 50:
        output = ResNet(x, BottleNeckBlock, [3, 4, 6, 3], class_dim)
    elif depth == 101:
        output = ResNet(x, BottleNeckBlock, [3, 4, 23, 3], class_dim)
    elif depth == 152:
        output = ResNet(x, BottleNeckBlock, [3, 8, 36, 3], class_dim)
    elif depth == 200:
        output = ResNet(x, BottleNeckBlock, [3, 24, 36, 3], class_dim)
    return output


def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out


def AttUNet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = conv_block(inputs, filters[0])

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = conv_block(e2, filters[1])

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = conv_block(e3, filters[2])

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = conv_block(e4, filters[3])

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = conv_block(e5, filters[4])

    d5 = up_conv(e5, filters[3])
    x4 = Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 = Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 = Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 = Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


class BilinearUpsampling(keras.layers.Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.upsampling = keras.utils.conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = keras.layers.InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        # .tf
        return tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                 int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling, 'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def xception_downsample_block(x, channels, top_relu=False):
    ##separable conv1
    if top_relu:
        x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv2
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_downsample_block(x, channels):
    res = Conv2D(channels, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = xception_downsample_block(x, channels)
    x = add([x, res])
    return x


def xception_block(x, channels):
    ##separable conv1
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv2
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv3
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_block(x, channels):
    res = x
    x = xception_block(x, channels)
    x = add([x, res])
    return x


def aspp(x, input_shape, out_stride):
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x


def DeeplabV3_plus(nClasses=21, input_height=512, input_width=512, out_stride=16):
    img_input = Input(shape=(input_height, input_width, 3))
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = res_xception_downsample_block(x, 128)

    res = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    skip = BatchNormalization()(x)
    x = Activation("relu")(skip)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = xception_downsample_block(x, 728, top_relu=True)

    for i in range(16):
        x = res_xception_block(x, 728)

    res = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(728, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(2048, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # aspp
    x = aspp(x, (input_height, input_width, 3), out_stride)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.9)(x)

    ##decoder
    x = BilinearUpsampling((4, 4))(x)
    dec_skip = Conv2D(48, (1, 1), padding="same", use_bias=False)(skip)
    dec_skip = BatchNormalization()(dec_skip)
    dec_skip = Activation("relu")(dec_skip)
    x = Concatenate()([x, dec_skip])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(nClasses, (1, 1), padding="same")(x)
    x = BilinearUpsampling((4, 4))(x)
    outputHeight = Model(img_input, x).output_shape[1]
    outputWidth = Model(img_input, x).output_shape[2]
    x = (Reshape((outputHeight * outputWidth, nClasses)))(x)
    x = Activation('softmax')(x)
    model = Model(input=img_input, output=x)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth
    return model


def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged

def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def en_build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    enet = BatchNormalization(momentum=0.1)(enet)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = PReLU(shared_axes=[1, 2])(enet)
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(4):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i

    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for _ in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8

    return enet

# decoder
def de_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder

def de_build(encoder, nc):
    enet = de_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = de_bottleneck(enet, 64)  # bottleneck 4.1
    enet = de_bottleneck(enet, 64)  # bottleneck 4.2
    enet = de_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = de_bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet

def ENet(n_classes, input_height=256, input_width=256):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    img_input = Input(shape=(input_height, input_width, 3))
    enet = en_build(img_input)
    enet = de_build(enet, n_classes)
    o_shape = Model(img_input, enet).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    enet = (Reshape((outputHeight*outputWidth, n_classes)))(enet)
    enet = Activation('softmax')(enet)
    model = Model(img_input, enet)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def ConvBnLayer(x, oup, kernel_size, stride, padding='valid'):
    y = Conv2D(filters=oup, kernel_size=kernel_size, strides=stride, padding=padding)(x)
    y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
    return y


def SELayer(x, reduction=4):
    batch, _, __, channel = x.shape
    y = GlobalAveragePooling2D()(x)
    y = Dense(units=channel.value // reduction, activation='relu')(y)
    y = Dense(units=channel.value, activation='sigmoid')(y)
    y = Reshape([1, 1, channel.value])(y)
    se_tensor = Multiply()([x, y])
    return se_tensor


def DepthWiseConv(x, kernel_size=3, stride=1, depth_multiplier=1, padding='same', relu=False):
    y = DepthwiseConv2D(kernel_size=kernel_size // 2, depth_multiplier=depth_multiplier,
                        strides=stride, padding=padding)(x)
    y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
    if relu:
        y = Activation('relu')(y)
    return y


def GhostModule(x, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
    init_channels = math.ceil(oup / ratio)
    new_channels = init_channels * (ratio - 1)

    multiplier = new_channels // init_channels

    primary_tensor = ConvBnLayer(x, init_channels, kernel_size=kernel_size, stride=stride, padding='same')
    if relu:
        primary_tensor = Activation('relu')(primary_tensor)

    cheap_tensor = DepthWiseConv(primary_tensor, kernel_size=dw_size,
                                 depth_multiplier=multiplier, padding='same', stride=1)
    if relu:
        cheap_tensor = Activation('relu')(cheap_tensor)

    out = Concatenate()([primary_tensor, cheap_tensor])
    return Lambda(lambda x: x[:, :, :, :oup])(out)


def GhostBottleneck(x, hidden_dim, oup, kernel_size, stride, use_se):
    assert stride in [1, 2]
    inp = x.shape[-1]
    if stride == 1 and inp == oup:
        shortcut = x
    else:
        shortcut = DepthWiseConv(x, kernel_size=3, stride=stride, relu=False)
        shortcut = ConvBnLayer(shortcut, oup, 1, 1, padding='same')

    x = GhostModule(x, hidden_dim, kernel_size=1, relu=True)
    if stride == 2:
        x = DepthWiseConv(x, kernel_size, stride, relu=False)
    if use_se:
        x = SELayer(x)
    x = GhostModule(x, oup, kernel_size=1, relu=False)
    return Add()([x, shortcut])


def GhostNet(x, num_classes=1000, width_mult=1.):
    cfgs = [
        # k, t, c, SE, s
        [3, 16, 16, 0, 1],
        [3, 48, 24, 0, 2],
        [3, 72, 24, 0, 1],
        [5, 72, 40, 1, 2],
        [5, 120, 40, 1, 1],
        [3, 240, 80, 0, 2],
        [3, 200, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]

    output_channel = _make_divisible(16 * width_mult, 4)

    x = ConvBnLayer(x, output_channel, 3, 2, padding='same')
    for k, exp_size, c, use_se, s in cfgs:
        output_channel = _make_divisible(c * width_mult, 4)
        hidden_channel = _make_divisible(exp_size * width_mult, 4)
        x = GhostBottleneck(x, hidden_channel, output_channel, k, s, use_se)

    output_channel = _make_divisible(exp_size * width_mult, 4)
    x = ConvBnLayer(x, output_channel, kernel_size=1, stride=1, padding='same')
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    output_channel = 1280
    x = Dense(output_channel)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes)(x)
    return x


def conv(x, outsize, kernel_size, strides_=1, padding_='same', activation=None):
    return Conv2D(outsize, kernel_size, strides=strides_, padding=padding_, kernel_initializer=RandomNormal(
        stddev=0.001), use_bias=False, activation=activation)(x)


def Bottleneck(x, size, downsampe=False):
    residual = x

    out = conv(x, size, 1, padding_='valid')
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size * 4, 1, padding_='valid')
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)

    if downsampe:
        residual = conv(x, size * 4, 1, padding_='valid')
        residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)

    out = Add()([out, residual])
    out = Activation('relu')(out)

    return out


def BasicBlock(x, size, downsampe=False):
    residual = x

    out = conv(x, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)
    out = Activation('relu')(out)

    out = conv(out, size, 3)
    out = BatchNormalization(epsilon=1e-5, momentum=0.1)(out)

    if downsampe:
        residual = conv(x, size, 1, padding_='valid')
        residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)

    out = Add()([out, residual])
    out = Activation('relu')(out)

    return out


def layer1(x):
    x = Bottleneck(x, 64, downsampe=True)
    x = Bottleneck(x, 64)
    x = Bottleneck(x, 64)
    x = Bottleneck(x, 64)

    return x


def transition_layer(x, in_channels, out_channels):
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []

    for i in range(num_out):
        if i < num_in:
            if in_channels[i] != out_channels[i]:
                residual = conv(x[i], out_channels[i], 3)
                residual = BatchNormalization(
                    epsilon=1e-5, momentum=0.1)(residual)
                residual = Activation('relu')(residual)
                out.append(residual)
            else:
                out.append(x[i])
        else:
            residual = conv(x[-1], out_channels[i], 3, strides_=2)
            residual = BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)
            residual = Activation('relu')(residual)
            out.append(residual)

    return out


def branches(x, block_num, channels):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = BasicBlock(residual, channels[i])
        out.append(residual)
    return out


def fuse_layers(x, channels, multi_scale_output=True):
    out = []

    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]
        for j in range(len(channels)):
            if j > i:
                y = conv(x[j], channels[i], 1, padding_='valid')
                y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                y = UpSampling2D(size=2 ** (j - i))(y)
                residual = Add()([residual, y])
            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = conv(y, channels[i], 3, strides_=2)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                    else:
                        y = conv(y, channels[j], 3, strides_=2)
                        y = BatchNormalization(epsilon=1e-5, momentum=0.1)(y)
                        y = Activation('relu')(y)
                residual = Add()([residual, y])

        residual = Activation('relu')(residual)
        out.append(residual)

    return out


def HighResolutionModule(x, channels, multi_scale_output=True):
    residual = branches(x, 4, channels)
    out = fuse_layers(residual, channels,
                      multi_scale_output=multi_scale_output)
    return out


def stage(x, num_modules, channels, multi_scale_output=True):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = HighResolutionModule(out, channels, multi_scale_output=False)
        else:
            out = HighResolutionModule(out, channels)

    return out


def HRNet(nClasses, input_height=224, input_width=224):
    channels_2 = [32, 64]
    channels_3 = [32, 64, 128]
    channels_4 = [32, 64, 128, 256]
    num_modules_2 = 1
    num_modules_3 = 4
    num_modules_4 = 3

    inputs = Input(shape=(input_height, input_width, 3))

    x = conv(inputs, 64, 3, strides_=2)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = conv(x, 64, 3, strides_=2)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)

    la1 = layer1(x)
    tr1 = transition_layer([la1], [256], channels_2)
    st2 = stage(tr1, num_modules_2, channels_2)
    tr2 = transition_layer(st2, channels_2, channels_3)
    st3 = stage(tr2, num_modules_3, channels_3)
    tr3 = transition_layer(st3, channels_3, channels_4)
    st4 = stage(tr3, num_modules_4, channels_4, multi_scale_output=False)
    up1 = UpSampling2D()(st4[0])
    up1 = conv(up1, 32, 3)
    up1 = BatchNormalization(epsilon=1e-5, momentum=0.1)(up1)
    up1 = Activation('relu')(up1)
    up2 = UpSampling2D()(up1)
    up2 = conv(up2, 32, 3)
    up2 = BatchNormalization(epsilon=1e-5, momentum=0.1)(up2)
    up2 = Activation('relu')(up2)

    final = conv(up2, nClasses, 1, padding_='valid')

    outputHeight = Model(inputs, final).output_shape[1]
    outputWidth = Model(inputs, final).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(final)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


def ICNet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))

    # (1/2)
    y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)),
               name='data_sub2')(inputs)
    y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_1_3x3_s2')(y)
    y = BatchNormalization(name='conv1_1_3x3_s2_bn')(y)
    y = Conv2D(32, 3, padding='same', activation='relu', name='conv1_2_3x3')(y)
    y = BatchNormalization(name='conv1_2_3x3_s2_bn')(y)
    y = Conv2D(64, 3, padding='same', activation='relu', name='conv1_3_3x3')(y)
    y = BatchNormalization(name='conv1_3_3x3_bn')(y)
    y_ = MaxPooling2D(pool_size=3, strides=2, name='pool1_3x3_s2')(y)

    y = Conv2D(128, 1, name='conv2_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv2_1_1x1_proj_bn')(y)
    y_ = Conv2D(32, 1, activation='relu', name='conv2_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv2_1_1x1_reduce_bn')(y_)
    y_ = ZeroPadding2D(name='padding1')(y_)
    y_ = Conv2D(32, 3, activation='relu', name='conv2_1_3x3')(y_)
    y_ = BatchNormalization(name='conv2_1_3x3_bn')(y_)
    y_ = Conv2D(128, 1, name='conv2_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv2_1_1x1_increase_bn')(y_)
    y = Add(name='conv2_1')([y, y_])
    y_ = Activation('relu', name='conv2_1/relu')(y)

    y = Conv2D(32, 1, activation='relu', name='conv2_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv2_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding2')(y)
    y = Conv2D(32, 3, activation='relu', name='conv2_2_3x3')(y)
    y = BatchNormalization(name='conv2_2_3x3_bn')(y)
    y = Conv2D(128, 1, name='conv2_2_1x1_increase')(y)
    y = BatchNormalization(name='conv2_2_1x1_increase_bn')(y)
    y = Add(name='conv2_2')([y, y_])
    y_ = Activation('relu', name='conv2_2/relu')(y)

    y = Conv2D(32, 1, activation='relu', name='conv2_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv2_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding3')(y)
    y = Conv2D(32, 3, activation='relu', name='conv2_3_3x3')(y)
    y = BatchNormalization(name='conv2_3_3x3_bn')(y)
    y = Conv2D(128, 1, name='conv2_3_1x1_increase')(y)
    y = BatchNormalization(name='conv2_3_1x1_increase_bn')(y)
    y = Add(name='conv2_3')([y, y_])
    y_ = Activation('relu', name='conv2_3/relu')(y)

    y = Conv2D(256, 1, strides=2, name='conv3_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv3_1_1x1_proj_bn')(y)
    y_ = Conv2D(64, 1, strides=2, activation='relu', name='conv3_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv3_1_1x1_reduce_bn')(y_)
    y_ = ZeroPadding2D(name='padding4')(y_)
    y_ = Conv2D(64, 3, activation='relu', name='conv3_1_3x3')(y_)
    y_ = BatchNormalization(name='conv3_1_3x3_bn')(y_)
    y_ = Conv2D(256, 1, name='conv3_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv3_1_1x1_increase_bn')(y_)
    y = Add(name='conv3_1')([y, y_])
    z = Activation('relu', name='conv3_1/relu')(y)

    # (1/4)
    y_ = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)),
                name='conv3_1_sub4')(z)
    y = Conv2D(64, 1, activation='relu', name='conv3_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding5')(y)
    y = Conv2D(64, 3, activation='relu', name='conv3_2_3x3')(y)
    y = BatchNormalization(name='conv3_2_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_2_1x1_increase')(y)
    y = BatchNormalization(name='conv3_2_1x1_increase_bn')(y)
    y = Add(name='conv3_2')([y, y_])
    y_ = Activation('relu', name='conv3_2/relu')(y)

    y = Conv2D(64, 1, activation='relu', name='conv3_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding6')(y)
    y = Conv2D(64, 3, activation='relu', name='conv3_3_3x3')(y)
    y = BatchNormalization(name='conv3_3_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_3_1x1_increase')(y)
    y = BatchNormalization(name='conv3_3_1x1_increase_bn')(y)
    y = Add(name='conv3_3')([y, y_])
    y_ = Activation('relu', name='conv3_3/relu')(y)

    y = Conv2D(64, 1, activation='relu', name='conv3_4_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_4_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding7')(y)
    y = Conv2D(64, 3, activation='relu', name='conv3_4_3x3')(y)
    y = BatchNormalization(name='conv3_4_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_4_1x1_increase')(y)
    y = BatchNormalization(name='conv3_4_1x1_increase_bn')(y)
    y = Add(name='conv3_4')([y, y_])
    y_ = Activation('relu', name='conv3_4/relu')(y)

    y = Conv2D(512, 1, name='conv4_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv4_1_1x1_proj_bn')(y)
    y_ = Conv2D(128, 1, activation='relu', name='conv4_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv4_1_1x1_reduce_bn')(y_)
    y_ = ZeroPadding2D(padding=2, name='padding8')(y_)
    y_ = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_1_3x3')(y_)
    y_ = BatchNormalization(name='conv4_1_3x3_bn')(y_)
    y_ = Conv2D(512, 1, name='conv4_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv4_1_1x1_increase_bn')(y_)
    y = Add(name='conv4_1')([y, y_])
    y_ = Activation('relu', name='conv4_1/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding9')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_2_3x3')(y)
    y = BatchNormalization(name='conv4_2_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_2_1x1_increase')(y)
    y = BatchNormalization(name='conv4_2_1x1_increase_bn')(y)
    y = Add(name='conv4_2')([y, y_])
    y_ = Activation('relu', name='conv4_2/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding10')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_3_3x3')(y)
    y = BatchNormalization(name='conv4_3_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_3_1x1_increase')(y)
    y = BatchNormalization(name='conv4_3_1x1_increase_bn')(y)
    y = Add(name='conv4_3')([y, y_])
    y_ = Activation('relu', name='conv4_3/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_4_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_4_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding11')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_4_3x3')(y)
    y = BatchNormalization(name='conv4_4_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_4_1x1_increase')(y)
    y = BatchNormalization(name='conv4_4_1x1_increase_bn')(y)
    y = Add(name='conv4_4')([y, y_])
    y_ = Activation('relu', name='conv4_4/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_5_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_5_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding12')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_5_3x3')(y)
    y = BatchNormalization(name='conv4_5_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_5_1x1_increase')(y)
    y = BatchNormalization(name='conv4_5_1x1_increase_bn')(y)
    y = Add(name='conv4_5')([y, y_])
    y_ = Activation('relu', name='conv4_5/relu')(y)

    y = Conv2D(128, 1, activation='relu', name='conv4_6_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_6_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding13')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_6_3x3')(y)
    y = BatchNormalization(name='conv4_6_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_6_1x1_increase')(y)
    y = BatchNormalization(name='conv4_6_1x1_increase_bn')(y)
    y = Add(name='conv4_6')([y, y_])
    y = Activation('relu', name='conv4_6/relu')(y)

    y_ = Conv2D(1024, 1, name='conv5_1_1x1_proj')(y)
    y_ = BatchNormalization(name='conv5_1_1x1_proj_bn')(y_)
    y = Conv2D(256, 1, activation='relu', name='conv5_1_1x1_reduce')(y)
    y = BatchNormalization(name='conv5_1_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding14')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_1_3x3')(y)
    y = BatchNormalization(name='conv5_1_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_1_1x1_increase')(y)
    y = BatchNormalization(name='conv5_1_1x1_increase_bn')(y)
    y = Add(name='conv5_1')([y, y_])
    y_ = Activation('relu', name='conv5_1/relu')(y)

    y = Conv2D(256, 1, activation='relu', name='conv5_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv5_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding15')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_2_3x3')(y)
    y = BatchNormalization(name='conv5_2_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_2_1x1_increase')(y)
    y = BatchNormalization(name='conv5_2_1x1_increase_bn')(y)
    y = Add(name='conv5_2')([y, y_])
    y_ = Activation('relu', name='conv5_2/relu')(y)

    y = Conv2D(256, 1, activation='relu', name='conv5_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv5_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding16')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_3_3x3')(y)
    y = BatchNormalization(name='conv5_3_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_3_1x1_increase')(y)
    y = BatchNormalization(name='conv5_3_1x1_increase_bn')(y)
    y = Add(name='conv5_3')([y, y_])
    y = Activation('relu', name='conv5_3/relu')(y)

    h, w = y.shape[1:3].as_list()
    pool1 = AveragePooling2D(pool_size=(h, w), strides=(h, w), name='conv5_3_pool1')(y)
    pool1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool1_interp')(pool1)
    pool2 = AveragePooling2D(pool_size=(h / 2, w / 2), strides=(h // 2, w // 2), name='conv5_3_pool2')(y)
    pool2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool2_interp')(pool2)
    pool3 = AveragePooling2D(pool_size=(h / 3, w / 3), strides=(h // 3, w // 3), name='conv5_3_pool3')(y)
    pool3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool3_interp')(pool3)
    pool6 = AveragePooling2D(pool_size=(h / 4, w / 4), strides=(h // 4, w // 4), name='conv5_3_pool6')(y)
    pool6 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool6_interp')(pool6)

    y = Add(name='conv5_3_sum')([y, pool1, pool2, pool3, pool6])
    y = Conv2D(256, 1, activation='relu', name='conv5_4_k1')(y)
    y = BatchNormalization(name='conv5_4_k1_bn')(y)
    aux_1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                   name='conv5_4_interp')(y)
    y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
    y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
    y = BatchNormalization(name='conv_sub4_bn')(y)
    y_ = Conv2D(128, 1, name='conv3_1_sub2_proj')(z)
    y_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_)
    y = Add(name='sub24_sum')([y, y_])
    y = Activation('relu', name='sub24_sum/relu')(y)

    aux_2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                   name='sub24_sum_interp')(y)
    y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
    y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
    y_ = BatchNormalization(name='conv_sub2_bn')(y_)

    # (1)
    y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_sub1')(inputs)
    y = BatchNormalization(name='conv1_sub1_bn')(y)
    y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv2_sub1')(y)
    y = BatchNormalization(name='conv2_sub1_bn')(y)
    y = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv3_sub1')(y)
    y = BatchNormalization(name='conv3_sub1_bn')(y)
    y = Conv2D(128, 1, name='conv3_sub1_proj')(y)
    y = BatchNormalization(name='conv3_sub1_proj_bn')(y)

    y = Add(name='sub12_sum')([y, y_])
    y = Activation('relu', name='sub12_sum/relu')(y)
    y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
               name='sub12_sum_interp')(y)

    o = Conv2D(nClasses, 1, name='conv6_cls')(y)

    o_shape = Model(inputs, o).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    o = (Activation('softmax'))(o)
    model = Model(inputs, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model


def relu6(x):
    return K.relu(x, max_value=6)

# Width Multiplier: Thinner Models
def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel, padding='valid', use_bias=False, strides=strides, name='conv1')(x)
    x = BatchNormalization(axis=3, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def MobileNetFCN8 (nClasses, optimizer=None, input_width=512, input_height=512,  pretrained='imagenet'):
    input_size = (input_height, input_width, 3)
    img_input = Input(input_size)
    alpha = 1.0
    depth_multiplier = 1
    x = conv_block(img_input, 16, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 16, alpha, depth_multiplier, block_id=1)
    f1 = x
    x = depthwise_conv_block(x, 32, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=3)
    f2 = x
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=5)
    f3 = x
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=11)
    f4 = x
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=13)
    f5 = x

    o = f5

    o = (Conv2D(256, (7, 7), activation='relu', padding='same'))(o)
    o = BatchNormalization()(o)


    o = (Conv2D(nClasses, (1, 1)))(o)
    # W = (N - 1) * S - 2P + F = 6 * 2 - 0 + 2 = 14
    o = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid")(o)
    # 14 x 14

    o2 = f4
    o2 = (Conv2D(nClasses, (1, 1)))(o2)

    # (14 x 14) (14 x 14)

    o = Add()([o, o2])
    # W = (N - 1) * S - 2P + F = 13 * 2 - 0 + 2 = 28
    o = Conv2DTranspose(nClasses, kernel_size=(2, 2),  strides=(2, 2), padding="valid")(o)
    o2 = f3
    o2 = (Conv2D(nClasses,  (1, 1)))(o2)
    # (28 x 28) (28 x 28)
    o = Add()([o2, o])

    # 224 x 224
    # W = (N - 1) * S - 2P + F = 27 * 8 + 8 = 224
    o = Conv2DTranspose(nClasses , kernel_size=(8,8),  strides=(8,8), padding="valid")(o)

    o_shape = Model(img_input, o).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model


def conv_block_nested(x, mid_ch, out_ch, kernel_size=3, padding='same'):
    x = Conv2D(mid_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(out_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def NestedUNet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    t = 2
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    x0_0 = conv_block_nested(inputs, filters[0], filters[0])

    x1_0 = conv_block_nested(MaxPooling2D(strides=2)(x0_0), filters[1], filters[1])
    x0_1 = conv_block_nested(Concatenate()([x0_0, UpSampling2D()(x1_0)]), filters[0], filters[0])

    x2_0 = conv_block_nested(MaxPooling2D(strides=2)(x1_0), filters[2], filters[2])
    x1_1 = conv_block_nested(Concatenate()([x1_0, UpSampling2D()(x2_0)]), filters[1], filters[1])
    x0_2 = conv_block_nested(Concatenate()([x0_0, x0_1, UpSampling2D()(x1_1)]), filters[0], filters[0])

    x3_0 = conv_block_nested(MaxPooling2D(strides=2)(x2_0), filters[3], filters[3])
    x2_1 = conv_block_nested(Concatenate()([x2_0, UpSampling2D()(x3_0)]), filters[2], filters[2])
    x1_2 = conv_block_nested(Concatenate()([x1_0, x1_1, UpSampling2D()(x2_1)]), filters[1], filters[1])
    x0_3 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, UpSampling2D()(x1_2)]), filters[0], filters[0])

    x4_0 = conv_block_nested(MaxPooling2D(strides=2)(x3_0), filters[4], filters[4])
    x3_1 = conv_block_nested(Concatenate()([x3_0, UpSampling2D()(x4_0)]), filters[3], filters[3])
    x2_2 = conv_block_nested(Concatenate()([x2_0, x2_1, UpSampling2D()(x3_1)]), filters[2], filters[2])
    x1_3 = conv_block_nested(Concatenate()([x1_0, x1_1, x1_2, UpSampling2D()(x2_2)]), filters[1], filters[1])
    x0_4 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, x0_3, UpSampling2D()(x1_3)]), filters[0], filters[0])

    o = Conv2D(nClasses, (3, 3), padding='same')(x0_4)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


def pool_block(inp, pool_factor):
    h = K.int_shape(inp)[1]
    w = K.int_shape(inp)[2]
    pool_size = strides = [int(np.round( float(h) / pool_factor)), int(np.round( float(w)/ pool_factor))]
    x = AveragePooling2D(pool_size, strides=strides, padding='same')(inp)
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*strides[0], int(x.shape[2])*strides[1])))(x)
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    return x


def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out

def Recurrent_block(input, channel, t=2):
    for i in range(t):
        if i == 0:
            x = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        out = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(add([x, x]))
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
    return out

def RRCNN_block(input, channel, t=2):
    x1 = Conv2D(channel, kernel_size=(1, 1), strides=1, padding='same')(input)
    x2 = Recurrent_block(x1, channel, t=t)
    x2 = Recurrent_block(x2, channel, t=t)
    out = add([x1, x2])
    return out



def R2AttUNet(nClasses, input_height=224, input_width=224):
    # """
    #Residual Recuurent Block with attention Unet
    #Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    #"""
    inputs = Input(shape=(input_height, input_width, 3))
    t = 2
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = RRCNN_block(inputs, filters[0], t=t)

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = RRCNN_block(e2, filters[1], t=t)

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = RRCNN_block(e3, filters[2], t=t)

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = RRCNN_block(e4, filters[3], t=t)

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = RRCNN_block(e5, filters[4], t=t)

    d5 = up_conv(e5, filters[3])
    x4 =  Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 =  Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 =  Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 =  Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


def SEModule(input, ratio, out_dim):
    # bs, c, h, w
    x = GlobalAveragePooling2D()(input)
    excitation = Dense(units=out_dim // ratio)(x)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)
    scale = multiply([input, excitation])
    return scale


def SEUnet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    # se
    conv1 = SEModule(conv1, 4, 16)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    # se
    conv2 = SEModule(conv2, 8, 32)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)

    # se
    conv3 = SEModule(conv3, 8, 64)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    # se
    conv4 = SEModule(conv4, 16, 128)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    # se
    conv5 = SEModule(conv5, 16, 256)

    up6 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv5))
    up6 = BatchNormalization()(up6)

    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    # se
    conv6 = SEModule(conv6, 16, 128)

    up7 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv6))
    up7 = BatchNormalization()(up7)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    # se
    conv7 = SEModule(conv7, 8, 64)

    up8 = Conv2D(32,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv7))
    up8 = BatchNormalization()(up8)

    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)

    conv8 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    # se
    conv8 = SEModule(conv8, 4, 32)

    up9 = Conv2D(16,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv8))
    up9 = BatchNormalization()(up9)

    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    # se
    conv9 = SEModule(conv9, 2, 16)

    conv10 = Conv2D(nClasses, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)

    outputHeight = Model(inputs, conv10).output_shape[1]
    outputWidth = Model(inputs, conv10).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(conv10)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x


def Unet_Xception_ResNetBlock(nClasses, input_height=224, input_width=224):
    from keras.applications.xception import Xception
    backbone = Xception(input_shape=(input_height, input_width, 3), weights=None, include_top=False)

    inputs = backbone.input

    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)

    # Middle
    convm = Conv2D(16 * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, 16 * 32)
    convm = residual_block(convm, 16 * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(16 * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.1)(uconv4)

    uconv4 = Conv2D(16 * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, 16 * 16)
    uconv4 = residual_block(uconv4, 16 * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(16 * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.1)(uconv3)

    uconv3 = Conv2D(16 * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, 16 * 8)
    uconv3 = residual_block(uconv3, 16 * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(16 * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1, 0), (1, 0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(16 * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, 16 * 4)
    uconv2 = residual_block(uconv2, 16 * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(16 * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3, 0), (3, 0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(16 * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, 16 * 2)
    uconv1 = residual_block(uconv1, 16 * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    # 128 -> 256
    uconv0 = Conv2DTranspose(16 * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(16 * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, 16 * 1)
    uconv0 = residual_block(uconv0, 16 * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(0.1 / 2)(uconv0)

    out = Conv2D(nClasses, (3, 3), padding='same')(uconv0)
    out = BatchNormalization()(out)

    outputHeight = Model(inputs, out).output_shape[1]
    outputWidth = Model(inputs, out).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(out)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    # mean = prevlayer
    # K.int_shape() Returns the shape of tensor or variable as a tuple of int or None entries
    lin1 = Dense(K.int_shape(prevlayer)[
                 3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[
                 3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x

# channel squeeze and spatial excitation


def sse_block(prevlayer, prefix):
    # Bug? Should be 1 here?
    conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal",
                  activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv

# concurrent spatial and channel squeeze and channel excitation


def csse_block(x, prefix):
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x


def scSEUnet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv5))
    up6 = BatchNormalization()(up6)

    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv6))
    up7 = BatchNormalization()(up7)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv7 = csse_block(conv7, prefix="conv7")

    up8 = Conv2D(32,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv7))
    up8 = BatchNormalization()(up8)

    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)

    conv8 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    conv8 = csse_block(conv8, prefix="conv8")

    up9 = Conv2D(16,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,
                                                                    2))(conv8))
    up9 = BatchNormalization()(up9)

    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(16,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv9 = csse_block(conv9, prefix="conv9")

    conv10 = Conv2D(nClasses, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)

    outputHeight = Model(inputs, conv10).output_shape[1]
    outputWidth = Model(inputs, conv10).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(conv10)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


# Model from https://github.com/BBuf/Keras-Semantic-Segmentation
class TestSemanticSegmentation(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_ResACNet(self):
        K.clear_session()
        input_shape = (224, 224, 3)
        inputs = Input(shape=input_shape, name="inputs")
        y = ResACNet(inputs, depth=50)
        keras_model = keras.models.Model(inputs=inputs, outputs=y)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_AttUNet(self):
        K.clear_session()
        keras_model = AttUNet(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DeepLabV3Plus(self):
        K.clear_session()
        keras_model = DeeplabV3_plus(input_height=224, input_width=224)
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    def test_ENet(self):
        K.clear_session()
        keras_model = ENet(80)
        data = np.random.rand(1, 256, 256, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    def test_GhostNet(self):
        K.clear_session()
        input_shape = (224, 224, 3)
        inputs = Input(shape=input_shape, name="inputs")
        y = GhostNet(inputs, 80)
        keras_model = keras.models.Model(inputs=inputs, outputs=y)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_HRNet(self):
        K.clear_session()
        keras_model = HRNet(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(get_maximum_opset_supported() < 10,
                     reason="Upsample op need scale value >= 1.0.")
    def test_ICNet(self):
        K.clear_session()
        keras_model = ICNet(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skip("TODO: perf.")
    def test_MobileNetFCN8(self):
        K.clear_session()
        keras_model = MobileNetFCN8(80)
        data = np.random.rand(2, 512, 512, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_NestedUNet(self):
        K.clear_session()
        keras_model = NestedUNet(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_R2AttUNet(self):
        K.clear_session()
        keras_model = R2AttUNet(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_SEUnet(self):
        K.clear_session()
        keras_model = SEUnet(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_Unet_Xception_ResNetBlock(self):
        K.clear_session()
        keras_model = Unet_Xception_ResNetBlock(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    def test_scSEUnet(self):
        K.clear_session()
        keras_model = scSEUnet(80)
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = mock_keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))

if __name__ == "__main__":
    unittest.main()
