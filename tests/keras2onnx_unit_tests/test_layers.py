# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from mock_keras2onnx.proto.tfcompat import is_tf2, tensorflow as tf
from mock_keras2onnx.proto import (keras, is_tf_keras,
                                   is_tensorflow_older_than, is_tensorflow_later_than,
                                   is_keras_older_than, is_keras_later_than, python_keras_is_deprecated)
from test_utils import no_loops_in_tf2, all_recurrents_should_bidirectional, convert_keras_for_test as convert_keras, get_max_opset_supported_for_test as get_maximum_opset_supported

K = keras.backend
Activation = keras.layers.Activation
Add = keras.layers.Add
if python_keras_is_deprecated():
    advanced_activations = keras.layers
    layers_core = keras.layers
else:
    advanced_activations = keras.layers.advanced_activations
    layers_core = keras.layers.core
AlphaDropout = keras.layers.AlphaDropout
Average = keras.layers.Average
AveragePooling1D = keras.layers.AveragePooling1D
AveragePooling2D = keras.layers.AveragePooling2D
AveragePooling3D = keras.layers.AveragePooling3D
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
Concatenate = keras.layers.Concatenate
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
Conv3D = keras.layers.Conv3D
Conv3DTranspose = keras.layers.Conv3DTranspose
Cropping1D = keras.layers.Cropping1D
Cropping2D = keras.layers.Cropping2D
Cropping3D = keras.layers.Cropping3D
Dense = keras.layers.Dense
Dot = keras.layers.Dot
dot = keras.layers.dot
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GaussianDropout = keras.layers.GaussianDropout
GaussianNoise = keras.layers.GaussianNoise
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
GRU = keras.layers.GRU
Input = keras.layers.Input
InputLayer = keras.layers.InputLayer
Lambda = keras.layers.Lambda
Layer = keras.layers.Layer
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
LSTMCell = keras.layers.LSTMCell
Maximum = keras.layers.Maximum
MaxPool1D = keras.layers.MaxPool1D
MaxPool3D = keras.layers.MaxPool3D
MaxPooling2D = keras.layers.MaxPooling2D
Model = keras.models.Model
Multiply = keras.layers.Multiply
Reshape = keras.layers.Reshape
RNN = keras.layers.RNN
SeparableConv1D = keras.layers.SeparableConv1D
SeparableConv2D = keras.layers.SeparableConv2D
Sequential = keras.models.Sequential
SimpleRNN = keras.layers.SimpleRNN
SpatialDropout2D = keras.layers.SpatialDropout2D
Subtract = keras.layers.Subtract
TimeDistributed = keras.layers.TimeDistributed
UpSampling1D = keras.layers.UpSampling1D
UpSampling2D = keras.layers.UpSampling2D
UpSampling3D = keras.layers.UpSampling3D
ZeroPadding2D = keras.layers.ZeroPadding2D
if not is_keras_older_than("2.2.4"):
    ReLU = keras.layers.ReLU

GRU_CLASSES = [(GRU, "v1")]
LSTM_CLASSES = [(LSTM, LSTMCell, "v1")]
RNN_CLASSES = [SimpleRNN, GRU, LSTM]

if is_tf_keras and is_tensorflow_later_than("1.14.0") and not python_keras_is_deprecated():
    # Add the TF v2 compatability layers (available after TF 1.14)
    from tensorflow.python.keras.layers import recurrent_v2
    GRU_CLASSES.append((recurrent_v2.GRU, "v2"))
    LSTM_CLASSES.append((recurrent_v2.LSTM, recurrent_v2.LSTMCell, "v2"))
    RNN_CLASSES.extend([recurrent_v2.GRU, recurrent_v2.LSTM])


def _asarray(*a):
    return np.array([a], dtype='f')


def test_keras_lambda(runner):
    model = Sequential()
    model.add(Lambda(lambda x: x ** 2, input_shape=[3, 5]))
    if get_maximum_opset_supported() >= 11:
        model.add(Lambda(lambda x: tf.round(x), input_shape=[3, 5]))
    model.add(Flatten(data_format='channels_last'))
    model.compile(optimizer='sgd', loss='mse')

    onnx_model = convert_keras(model, 'test_keras_lambda')
    data = np.random.rand(3 * 5).astype(np.float32).reshape(1, 3, 5)
    expected = model.predict(data)
    assert runner('onnx_lambda', onnx_model, data, expected)


@pytest.mark.skipif(is_tensorflow_older_than('1.12.0'),
                    reason="tf.nn.depth_to_space not supported.")
@pytest.mark.skipif(get_maximum_opset_supported() < 11,
                    reason="DepthToSpace is not supported before opset 11.")
@pytest.mark.parametrize("data_format", ["NCHW", "NHWC"])
@pytest.mark.parametrize("input_shape", [(4, 6, 8), (None, None, 8)])
def test_keras_lambda_depth_to_space(runner, data_format, input_shape):
    if data_format == "NCHW" and (is_tensorflow_older_than("2.2.0") or is_tensorflow_later_than("2.4.0")):
        pytest.skip("tf.nn.depth_to_space with NCHW only supported for Tensorflow 2.2 and 2.3")
    model = Sequential()
    model.add(Lambda(
        lambda x: tf.nn.depth_to_space(x, block_size=2, data_format=data_format),
        input_shape=input_shape
    ))

    onnx_model = convert_keras(model, 'test_keras_lambda_depth_to_space')
    data = np.random.rand(3, 4, 6, 8).astype(np.float32)  # batch dimension + 'input_shape'
    expected = model.predict(data)
    assert runner('tf_depth_to_space', onnx_model, data, expected)


def test_tf_addn(runner):
    input1 = Input(shape=(5, 3, 4), dtype=tf.float32)
    input2 = Input(shape=(5, 3, 4), dtype=tf.float32)
    sum = Lambda(tf.add_n)([input1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=sum)

    onnx_model = convert_keras(model, 'tf_add_n')
    batch_data1_shape = (2, 5, 3, 4)
    batch_data2_shape = (2, 5, 3, 4)
    data1 = np.random.rand(*batch_data1_shape).astype(np.float32)
    data2 = np.random.rand(*batch_data2_shape).astype(np.float32)
    expected = model.predict([data1, data2])
    assert runner('tf_add_n', onnx_model, [data1, data2], expected)


@pytest.mark.parametrize("arg_func", [tf.argmax, tf.argmin])
def test_tf_argmax_argmin(runner, arg_func):
    model = Sequential()
    model.add(Lambda(lambda x: arg_func(x, axis=2), input_shape=[3, 4, 2]))
    onnx_model = convert_keras(model, 'test_tf_arg')
    data = np.random.rand(5, 3, 4, 2).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_arg', onnx_model, data, expected)

    model = Sequential()
    model.add(Lambda(lambda x: arg_func(x, axis=2, output_type=tf.int32), input_shape=[3, 4, 2]))
    onnx_model = convert_keras(model, 'test_tf_arg')
    data = np.random.rand(5, 3, 4, 2).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_arg', onnx_model, data, expected)


@pytest.mark.parametrize("arg_func", [[tf.nn.avg_pool, tf.nn.avg_pool3d], [tf.nn.max_pool, tf.nn.max_pool3d]])
@pytest.mark.parametrize("padding_method", ['SAME', 'VALID'])
def test_tf_pool(runner, arg_func, padding_method):
    model = Sequential()
    k_size = [1, 2, 2, 1] if is_tensorflow_older_than('1.14.0') else 2
    model.add(Lambda(lambda x: arg_func[0](x, k_size, strides=[1, 1, 2, 1], padding=padding_method, data_format='NHWC'),
                     input_shape=[10, 12, 3]))
    onnx_model = convert_keras(model, 'test_tf_pool')
    data = np.random.rand(5, 10, 12, 3).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_pool2d', onnx_model, data, expected)

    model = Sequential()
    strides = [1, 1, 1, 2, 1] if is_tensorflow_older_than('1.14.0') else [1, 1, 2]
    k_size = [1, 2, 2, 2, 1] if is_tensorflow_older_than('1.14.0') else 2
    model.add(Lambda(lambda x: arg_func[1](x, k_size, strides=strides, padding=padding_method, data_format='NDHWC'),
                     input_shape=[10, 12, 3, 4]))
    onnx_model = convert_keras(model, 'test_tf_pool')
    data = np.random.rand(5, 10, 12, 3, 4).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_pool3d', onnx_model, data, expected)


def test_tf_conv(runner):
    model = Sequential()
    k = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(1, 2, 3, 5)).astype(np.float32))
    model.add(Lambda(lambda x: tf.nn.conv2d(x, k, strides=[1, 1, 2, 1], padding='SAME', data_format='NHWC'),
                     input_shape=[10, 14, 3]))
    onnx_model = convert_keras(model, 'test_tf_conv2d')
    data = np.random.rand(1, 10, 14, 3).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_conv2d', onnx_model, data, expected)

    model = Sequential()
    k = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(1, 2, 3, 5)).astype(np.float32))
    model.add(Lambda(lambda x: tf.nn.conv2d(x, k, strides=[1, 1, 2, 1], padding='VALID', data_format='NHWC'),
                     input_shape=[10, 14, 3]))
    onnx_model = convert_keras(model, 'test_tf_conv2d')
    data = np.random.rand(1, 10, 14, 3).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_conv2d', onnx_model, data, expected)

    model = Sequential()
    k = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(1, 3, 5)).astype(np.float32))
    model.add(Lambda(lambda x: tf.nn.conv1d(x, k, stride=2, padding='SAME', data_format='NWC'),
                     input_shape=[10, 3]))
    onnx_model = convert_keras(model, 'test_tf_conv1d')
    data = np.random.rand(1, 10, 3).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_conv1d', onnx_model, data, expected)

    model = Sequential()
    k = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(1, 2, 3, 5, 2)).astype(np.float32))
    model.add(Lambda(lambda x: tf.nn.conv3d(x, k, strides=[1, 1, 2, 1, 1], padding='SAME', data_format='NDHWC'),
                     input_shape=[10, 14, 3, 5]))
    onnx_model = convert_keras(model, 'test_tf_conv3d')
    data = np.random.rand(1, 10, 14, 3, 5).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_conv3d', onnx_model, data, expected)


@pytest.mark.skipif(is_tensorflow_older_than('1.14.0'),
                    reason="tf.math has no attribute 'floormod'.")
def test_tf_floormod(runner):
    def my_func_1(x):
        return tf.math.floormod(x[0], x[1])

    def my_func_2(x):
        return tf.math.floormod(tf.cast(x[0], tf.int32), tf.cast(x[1], tf.int32))

    for my_func_ in [my_func_1, my_func_2]:
        input1 = Input(shape=[2, 2])
        input2 = Input(shape=[2, 2])
        added = Lambda(my_func_)([input1, input2])
        model = keras.models.Model(inputs=[input1, input2], outputs=added)
        onnx_model = convert_keras(model, 'test_tf_floormod')
        data1 = 100 * np.random.rand(2, 2, 2).astype(np.float32) + 1.0
        data2 = 10 * np.random.rand(2, 2, 2).astype(np.float32) + 1.0
        expected = model.predict([data1, data2])
        assert runner('onnx_tf_floormod', onnx_model, [data1, data2], expected)


def test_tf_rsqrt(runner):
    def my_func_1(x):
        beta = tf.constant([0.0, 0.0, 0.0, 0.0])
        gamma = tf.constant([0.0, 0.0, 0.0, 0.0])
        mean = tf.constant([0.0, 0.0, 0.0, 0.0])
        variance = tf.constant([1.0, 1.0, 1.0, 1.0])
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)

    model = Sequential()
    model.add(Lambda(lambda x: my_func_1(x), input_shape=[2, 3, 4]))
    onnx_model = convert_keras(model, 'test_tf_rsqrt')
    data = np.random.rand(1, 2, 3, 4).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_rsqrt', onnx_model, data, expected)


def test_tf_bias_add(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.nn.bias_add(x, tf.constant([100., -100.])), input_shape=[3, 4, 2]))
    onnx_model = convert_keras(model, 'test_tf_bias_add')
    data = np.random.rand(5, 3, 4, 2).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_bias_add', onnx_model, data, expected)

    model = Sequential()
    model.add(
        Lambda(lambda x: tf.nn.bias_add(x, tf.constant([100., -100.]), data_format='NCHW'), input_shape=[2, 3, 4]))
    onnx_model = convert_keras(model, 'test_tf_bias_add')
    data = np.random.rand(5, 2, 3, 4).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_bias_add', onnx_model, data, expected)


def test_tf_clip(runner):
    model = Sequential()
    model.add(Lambda(lambda x: K.clip(x, 0, 10), input_shape=[5, 5]))
    data = np.random.randint(-5, 15, size=(1, 5, 5)).astype(np.float32)
    expected = model.predict(data)
    onnx_model = convert_keras(model, 'test_tf_clip')
    assert runner('onnx_tf_clip', onnx_model, data, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 12,
                    reason="Result mismatch on ORT, skip conversion for unsupported types.")
def test_tf_pow(runner):
    model = Sequential()
    y = tf.constant([[2.0, 2.0], [2.0, 2.0]])
    model.add(Lambda(lambda x: tf.math.pow(tf.cast(x, tf.int32), tf.cast(y, tf.int32)), input_shape=[2, 2]))
    data = (100 * np.random.rand(3, 2, 2)).astype(np.float32)
    expected = model.predict(data)
    onnx_model = convert_keras(model, 'test_tf_pow')
    assert runner('onnx_tf_pow', onnx_model, data, expected)


def test_tf_concat(runner):
    def my_func_1(x):
        return tf.concat([x[0], x[1]], 1)

    def my_func_2(x):
        return tf.concat([x[0], x[1]], -1)

    input1_shape = [(2, 3), (3, 2)]
    input2_shape = [(4, 3), (3, 4)]
    myFunc = [my_func_1, my_func_2]
    for idx_ in range(2):
        input1 = Input(shape=input1_shape[idx_])
        input2 = Input(shape=input2_shape[idx_])
        added = Lambda(myFunc[idx_])([input1, input2])
        model = keras.models.Model(inputs=[input1, input2], outputs=added)

        onnx_model = convert_keras(model, 'test_tf_concat')
        batch_data1_shape = (2,) + input1_shape[idx_]
        batch_data2_shape = (2,) + input2_shape[idx_]
        data1 = np.random.rand(*batch_data1_shape).astype(np.float32)
        data2 = np.random.rand(*batch_data2_shape).astype(np.float32)
        expected = model.predict([data1, data2])
        assert runner('onnx_concat', onnx_model, [data1, data2], expected)


@pytest.mark.parametrize("use_bias", [True, False])
def test_depthwise_conv2d(runner, use_bias):
    model = Sequential()
    model.add(InputLayer(input_shape=(8, 8, 2)))
    model.add(keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3), strides=(1, 1), padding="VALID",
        data_format='channels_last', use_bias=use_bias))
    onnx_model = convert_keras(model, 'test_depthwise_conv2d')
    data = np.random.rand(3, 8, 8, 2).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_depthwise_conv2d', onnx_model, data, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 12,
                    reason="Einsum is not supported until opset 12.")
def test_tf_einsum(runner):
    def my_func_1(x):
        return tf.einsum('i,d->id', x[0][:, 0], x[1][:, 1])

    def my_func_2(x):
        return tf.einsum('ibh,hnd->ibnd', x[0][:, 0], x[1][:, 1])

    def my_func_3(x):
        return tf.einsum('ibnd,hnd->ibh', x[0][:, 0], x[1][:, 1])

    def my_func_4(x):
        return tf.einsum('ibnd,jbnd->ijbn', x[0][:, 0], x[1][:, 1])

    def my_func_5(x):
        return tf.einsum('ijbn,jbnd->ibnd', x[0][:, 0], x[1][:, 1])

    input1_shape = [(3,), (2, 3, 2), (2, 3, 4, 2), (2, 3, 4, 2), (2, 2, 4, 3)]
    input2_shape = [(3,), (2, 4, 5), (2, 4, 2), (2, 3, 4, 2), (2, 4, 3, 5)]
    myFunc = [my_func_1, my_func_2, my_func_3, my_func_4, my_func_5]

    for idx_ in range(len(myFunc)):
        K.clear_session()
        input1 = Input(shape=input1_shape[idx_])
        input2 = Input(shape=input2_shape[idx_])
        added = Lambda(myFunc[idx_])([input1, input2])
        model = keras.models.Model(inputs=[input1, input2], outputs=added)
        onnx_model = convert_keras(model, 'test_tf_einsum')
        batch_data1_shape = (2,) + input1_shape[idx_]
        batch_data2_shape = (2,) + input2_shape[idx_]
        data1 = np.random.rand(*batch_data1_shape).astype(np.float32)
        data2 = np.random.rand(*batch_data2_shape).astype(np.float32)
        expected = model.predict([data1, data2])
        assert runner('onnx_einsum', onnx_model, [data1, data2], expected)


def test_tf_expand_dims(runner):
    for dim in [0, 1, -1]:
        model = Sequential()
        model.add(Lambda(lambda x: tf.expand_dims(x, dim), input_shape=[2, 3, 4]))
        onnx_model = convert_keras(model, 'test_tf_expand_dims')
        data = np.random.rand(1, 2, 3, 4).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_tf_expand_dims', onnx_model, data, expected)


def test_tf_fill(runner):
    model = Sequential()
    model.add(Lambda(lambda x: x + tf.fill([2, 3], 2.3), input_shape=[2, 3]))
    onnx_model = convert_keras(model, 'test_tf_fill')
    data = np.random.rand(3, 2, 3).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_fill', onnx_model, data, expected)


def test_tf_fused_batch_norm(runner):
    def my_func_1(x):
        beta = tf.constant([0.2, 0.3, 0.4, 0.5])
        gamma = tf.constant([0.5, 0.4, 0.3, 0.2])
        mean = tf.constant([0.1, 0.2, 0.3, 0.4])
        variance = tf.constant([0.9, 1.0, 1.0, 1.1])
        return tf.nn.fused_batch_norm(x, mean, variance, beta, gamma, 0.001, data_format='NHWC', is_training=False)[
            0]

    def my_func_2(x):
        beta = tf.constant([0.2, 0.3])
        gamma = tf.constant([0.5, 0.4])
        mean = tf.constant([0.1, 0.2])
        variance = tf.constant([0.9, 1.0])
        return tf.nn.fused_batch_norm(x, mean, variance, beta, gamma, 0.001, data_format='NCHW', is_training=False)[
            0]

    for my_func in [my_func_1, my_func_2]:
        model = Sequential()
        model.add(Lambda(lambda x: my_func(x), input_shape=[2, 3, 4]))
        onnx_model = convert_keras(model, 'test_tf_fused_batch_norm')
        data = np.random.rand(1, 2, 3, 4).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_tf_fused_batch_norm', onnx_model, data, expected)


def test_tf_gather(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.gather(x, [1, 1], axis=1), input_shape=[5, 5]))
    onnx_model = convert_keras(model, 'test_tf_gather')
    data = np.random.rand(3, 5, 5).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_gather', onnx_model, data, expected)


def test_tf_maximum_minimum(runner):
    input1_shape_list = [(2, 3), (2, 3)]
    input2_shape_list = [(2, 3), (2, 1)]

    def my_func_1(x):
        return tf.minimum(tf.maximum(x[0], x[1]), 0.5)

    def my_func_2(x):
        return tf.minimum(tf.maximum(x[0], 0.5), x[1])

    for idx_ in range(len(input1_shape_list)):
        for myFunc in [my_func_1, my_func_2]:
            input1 = Input(shape=input1_shape_list[idx_], dtype=tf.float32)
            input2 = Input(shape=input2_shape_list[idx_], dtype=tf.float32)
            added = Lambda(myFunc)([input1, input2])
            model = keras.models.Model(inputs=[input1, input2], outputs=added)

            onnx_model = convert_keras(model, 'tf_maximum_minimum')
            batch_data1_shape = (2,) + input1_shape_list[idx_]
            batch_data2_shape = (2,) + input2_shape_list[idx_]
            data1 = np.random.rand(*batch_data1_shape).astype(np.float32)
            data2 = np.random.rand(*batch_data2_shape).astype(np.float32)
            expected = model.predict([data1, data2])
            assert runner('tf_maximum_minimum', onnx_model, [data1, data2], expected)

    def my_func_3(x):
        return tf.minimum(tf.maximum(x[0], x[1]), 50)

    def my_func_4(x):
        return tf.minimum(tf.maximum(x[0], 50), x[1])

    for idx_ in range(len(input1_shape_list)):
        for myFunc in [my_func_3, my_func_4]:
            input1 = Input(shape=input1_shape_list[idx_], dtype=tf.int32)
            input2 = Input(shape=input2_shape_list[idx_], dtype=tf.int32)
            added = Lambda(myFunc)([input1, input2])
            model = keras.models.Model(inputs=[input1, input2], outputs=added)

            onnx_model = convert_keras(model, 'tf_maximum_minimum')
            batch_data1_shape = (2,) + input1_shape_list[idx_]
            batch_data2_shape = (2,) + input2_shape_list[idx_]
            data1 = (100 * np.random.rand(*batch_data1_shape)).astype(np.int32)
            data2 = (100 * np.random.rand(*batch_data2_shape)).astype(np.int32)
            expected = model.predict([data1, data2])
            assert runner('tf_maximum_minimum', onnx_model, [data1, data2], expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 9,
                    reason="opset < 9 is not supported.")
def test_tf_one_hot(runner):
    def my_func(x):
        return tf.one_hot(tf.cast(x, tf.int32), 3, 5.0, -1.0, 1)

    model = Sequential()
    model.add(Lambda(lambda x: my_func(x), input_shape=[3]))
    onnx_model = convert_keras(model, 'test_tf_one_hot')
    data = np.array([[0, 1, 2]]).astype(np.float32)
    expected = model.predict(data)
    assert runner('tf_one_hot', onnx_model, data, expected)


def test_tf_pad(runner):
    def my_func_1(x):
        paddings = tf.constant([[0, 0], [1, 3], [2, 4]])
        return tf.pad(x, paddings, mode='CONSTANT')

    def my_func_2(x):
        paddings = tf.constant([[0, 0], [1, 3], [2, 4]])
        return tf.pad(x, paddings, mode='CONSTANT', constant_values=1)

    def my_func_3(x):
        paddings = tf.constant([[0, 0], [1, 3], [2, 4]])
        return tf.pad(x, paddings, mode='REFLECT')

    for my_func in [my_func_1, my_func_2, my_func_3]:
        model = Sequential()
        model.add(Lambda(lambda x: my_func(x), input_shape=[5, 5]))
        onnx_model = convert_keras(model, 'test_tf_pad')
        data = np.random.rand(2, 5, 5).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_tf_pad', onnx_model, data, expected)


def test_tf_range(runner):
    def my_func_1(x):
        return x + tf.cast(tf.range(3, 18, 3), tf.float32)

    def my_func_2(x):
        return x + tf.range(2.3, 4.6, 0.8, dtype=tf.float32)

    for my_func_ in [my_func_1, my_func_2]:
        K.clear_session()
        model = Sequential()
        model.add(Lambda(lambda x: my_func_(x), input_shape=[1]))
        onnx_model = convert_keras(model, 'test_tf_range')
        data = np.random.rand(3, 1).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_range_1', onnx_model, data, expected)

    def my_func_3(x):
        return x[0] + tf.cast(tf.range(3, 18, tf.cast(x[1][0, 0], tf.int32)), tf.float32)

    K.clear_session()
    input1 = Input(shape=(5,))
    input2 = Input(shape=(1,))
    added = Lambda(my_func_3)([input1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=added)
    onnx_model = convert_keras(model, 'test_tf_range')
    data_1 = np.random.randint(1, 3, size=(1, 5)).astype(np.float32)
    data_2 = np.array([3]).astype(np.float32).reshape(1, 1)
    expected = model.predict([data_1, data_2])
    assert runner('onnx_range_2', onnx_model, [data_1, data_2], expected)


def test_tf_compare_equal(runner):
    for tf_op_ in [tf.not_equal, tf.less_equal, tf.greater_equal]:
        input1_shape = [[3], [3]]
        input1 = Input(shape=input1_shape[0], dtype='int32')
        input2 = Input(shape=input1_shape[1], dtype='int32')
        comp = Lambda(lambda x: tf_op_(x[0], x[1]))([input1, input2])
        model = keras.models.Model(inputs=[input1, input2], outputs=comp)

        onnx_model = convert_keras(model, 'tf_compare_equal')
        data1 = np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int32)
        data2 = np.array([[1, 2, 3], [2, 1, 4]]).astype(np.int32)
        expected = model.predict([data1, data2])
        assert runner('tf_compare_equal', onnx_model, [data1, data2], expected)


def test_tf_realdiv(runner):
    input1_shape = [(2, 3), (2, 3)]
    input2_shape = [(2, 3), (3,)]
    for idx_ in range(2):
        input1 = Input(shape=input1_shape[idx_])
        input2 = Input(shape=input2_shape[idx_])
        added = Lambda(lambda x: tf.realdiv(x[0], x[1]))([input1, input2])
        model = keras.models.Model(inputs=[input1, input2], outputs=added, name='realdiv')

        onnx_model = convert_keras(model, model.name)
        batch_data1_shape = (2,) + input1_shape[idx_]
        batch_data2_shape = (2,) + input2_shape[idx_]
        data1 = np.random.rand(*batch_data1_shape).astype(np.float32)
        data2 = np.random.rand(*batch_data2_shape).astype(np.float32)
        expected = model.predict([data1, data2])
        assert runner(model.name, onnx_model, [data1, data2], expected)


def test_tf_reduce_op(runner):
    reduce_name = ['tf_min', 'tf_max', 'tf_mean', 'tf_sum', 'tf_prod']
    reduce_ops = [K.min, K.max, K.mean, K.sum, K.prod]
    axis_list = [1] if is_tf_keras else [1, None]
    keepdims_val = [True] if is_tf_keras else [True, False]
    for idx, reduce_op in enumerate(reduce_ops):
        for axis in axis_list:
            for keepdims in keepdims_val:
                model = Sequential()
                model.add(Lambda(lambda x: reduce_op(x, axis=axis, keepdims=keepdims), input_shape=[2, 2]))
                onnx_model = convert_keras(model, 'test_' + reduce_name[idx])
                data = np.random.rand(3, 2, 2).astype(np.float32)
                expected = model.predict(data)
                assert runner('onnx_' + reduce_name[idx], onnx_model, data, expected)

    axis_list = [1] if is_tf2 and is_tf_keras else [1, None]
    for idx, reduce_op in enumerate(reduce_ops):
        for axis in axis_list:
            for keepdims in keepdims_val:
                model = Sequential()
                model.add(Lambda(lambda x: reduce_op(x, axis=axis, keepdims=keepdims), input_shape=[2, 2]))
                onnx_model = convert_keras(model, 'test_' + reduce_name[idx])
                data = np.random.rand(1, 2, 2).astype(np.float32)
                expected = model.predict(data)
                assert runner('onnx_' + reduce_name[idx], onnx_model, data, expected)


def test_tf_reshape(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.reshape(x, [-1, 2, 4]), input_shape=[2, 2, 2]))
    onnx_model = convert_keras(model, 'test_tf_reshape_float')
    data = np.random.rand(3, 2, 2, 2).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_reshape_float', onnx_model, data, expected)

    model = Sequential()
    model.add(Lambda(lambda x: tf.reshape(x, [-1, 2, 4]), input_shape=[2, 2, 2], dtype=tf.int32))
    onnx_model = convert_keras(model, 'test_tf_reshape_int')
    data = np.random.randint(5, size=(3, 2, 2, 2)).astype(np.int32)
    expected = model.predict(data)
    assert runner('onnx_reshape_int', onnx_model, data, expected)

    def my_func(x):
        return tf.reshape(x[0][0], tf.cast(x[1][0], tf.int32))

    input1 = Input(shape=(6,))
    input2 = Input(shape=(3,))
    added = Lambda(my_func)([input1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=added)
    onnx_model = convert_keras(model, 'test_tf_reshape_dynamic')
    data_1 = np.random.rand(1, 6).astype(np.float32).reshape(1, 6)
    data_2 = np.array([1, 2, 3]).astype(np.float32).reshape(1, 3)
    expected = model.predict([data_1, data_2])
    assert runner('onnx_reshape_dynamic', onnx_model, [data_1, data_2], expected)


def test_tf_resize(runner):
    target_opset = get_maximum_opset_supported()
    shape_list = [10, None] if target_opset >= 10 else [10]
    size_list = [[5, 10], [20, 30]] if target_opset >= 10 else [[20, 30]]
    for g in [tf.image.resize_bilinear, tf.image.resize_nearest_neighbor]:
        for shape_1_dim in shape_list:
            for size in size_list:
                model = Sequential()
                model.add(Lambda(lambda x: g(x, size=size), input_shape=[shape_1_dim, 20, 3]))

                onnx_model = convert_keras(model, 'test_tf_resize', target_opset=target_opset)
                data = np.random.rand(2, 10, 20, 3).astype(np.float32)
                expected = model.predict(data)
                assert runner('onnx_resize', onnx_model, data, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 11,
                    reason="Resize coordinate_transformation_mode need opset >= 11.")
@pytest.mark.skipif(is_tensorflow_older_than('1.14.0') or (is_tf2 and is_tensorflow_older_than('2.2.0')),
                    reason="module 'tensorflow.compat' has no attribute 'v1'.")
def test_tf_resize_2(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.compat.v1.image.resize(x,
                                                         [3, 4],
                                                         method='bilinear', align_corners=True),
                     input_shape=[5, 7, 3]))

    onnx_model = convert_keras(model, 'test_tf_resize_2')
    data = np.random.rand(2, 5, 7, 3).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_resize_2', onnx_model, data, expected)


def test_tf_size(runner):
    model = Sequential()
    model.add(Lambda(lambda x: x + tf.cast(tf.size(x), tf.float32), input_shape=[2, 3, 5]))
    onnx_model = convert_keras(model, 'test_tf_size')
    data = np.random.rand(3, 2, 3, 5).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_size', onnx_model, data, expected)


def test_tf_slice(runner):
    model = Sequential()
    # Need 0th: start=0 size=batch_dim
    model.add(Lambda(lambda x: tf.slice(x, [0, 1, 0, 2], [3, 1, 2, 2]), input_shape=[2, 3, 5]))
    onnx_model = convert_keras(model, 'test_tf_slice')
    data = np.random.rand(3, 2, 3, 5).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_slice', onnx_model, data, expected)

    if get_maximum_opset_supported() < 10:
        return

    def my_func_1(x):
        return tf.slice(x[0], tf.cast(x[1][0], tf.int32), [3, 1, 1, 2])

    input1 = Input(shape=(2, 3, 5), name='inputs')
    input2 = Input(shape=(4,), dtype=tf.int32, name='begin')
    added = Lambda(my_func_1)([input1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=added)
    onnx_model = convert_keras(model, 'test_tf_slice')
    data1 = np.random.rand(3, 2, 3, 5).astype(np.float32)
    data2 = np.array([[0, 1, 0, 2], [0, 1, 0, 2], [0, 1, 0, 2]]).astype(np.int32)
    expected = model.predict([data1, data2])
    assert runner('onnx_tf_slice', onnx_model, {"inputs": data1, 'begin': data2}, expected)

    def my_func_2(x):
        return tf.slice(x[0], [0, 1, 0, 2], tf.cast(x[1][0], tf.int32))

    input1 = Input(shape=(2, 3, 5), name='inputs')
    input2 = Input(shape=(4,), dtype=tf.int32, name='size')
    added = Lambda(my_func_2)([input1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=added)
    onnx_model = convert_keras(model, 'test_tf_slice')
    data1 = np.random.rand(3, 2, 3, 5).astype(np.float32)
    data2 = np.array([[3, 1, 1, 2], [3, 1, 1, 2], [3, 1, 1, 2]]).astype(np.int32)
    expected = model.predict([data1, data2])
    assert runner('onnx_tf_slice', onnx_model, {"inputs": data1, 'size': data2}, expected)


def test_tf_softmax(runner):
    for func_ in [lambda x: tf.nn.softmax(x), lambda x: tf.nn.softmax(x, axis=-1), lambda x: tf.nn.softmax(x, axis=1)]:
        model = Sequential()
        model.add(Lambda(func_, input_shape=[2, 3, 5]))
        onnx_model = convert_keras(model, 'test_tf_softmax')
        data = np.random.rand(3, 2, 3, 5).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_tf_softmax', onnx_model, data, expected)


def test_tf_sqrt(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.sqrt(x + 1.0), input_shape=[2, 5]))
    onnx_model = convert_keras(model, 'test_tf_sqrt')
    data = np.random.rand(3, 2, 5).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_sqrt', onnx_model, data, expected)


@pytest.mark.skipif(is_tensorflow_older_than('1.14.0'),
                    reason="dilations in tf.nn.depthwise_conv2d not supported.")
def test_tf_space_to_batch_nd(runner):
    model = Sequential()
    filter_value = np.random.rand(3, 3, 2, 2).astype(np.float32)
    filter_constant = tf.constant(filter_value.tolist(), dtype=tf.float32)
    model.add(Lambda(lambda x: tf.nn.depthwise_conv2d(
        x, filter=filter_constant, strides=(1, 1, 1, 1), padding="VALID",
        data_format='NHWC', dilations=(2, 2)), input_shape=(8, 8, 2)))
    onnx_model = convert_keras(model, 'test_tf_space_to_batch_nd')
    data = np.random.rand(3, 8, 8, 2).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_space_to_batch_nd', onnx_model, data, expected)


def test_tf_splitv(runner):
    def my_func_1(x):
        return tf.split(x, [4, 15, 11], 2)[0]

    model = Sequential()
    model.add(Lambda(lambda x: my_func_1(x), input_shape=[5, 30]))
    onnx_model = convert_keras(model, 'test_tf_splitv')
    data = np.random.rand(2, 5, 30).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_splitv', onnx_model, data, expected)


def test_tf_square(runner):
    model = Sequential()
    model.add(Lambda(lambda x: x + tf.square(x), input_shape=[2, 3, 5]))
    onnx_model = convert_keras(model, 'test_tf_square')
    data = np.random.rand(3, 2, 3, 5).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tf_square', onnx_model, data, expected)


def test_tf_squeeze(runner):
    for func_ in [lambda x: tf.squeeze(x, [1]), lambda x: tf.squeeze(x), lambda x: tf.squeeze(x, [-2])]:
        model = Sequential()
        model.add(Lambda(func_, input_shape=[1, 2, 1, 2]))
        onnx_model = convert_keras(model, 'test_tf_squeeze')
        data = np.random.rand(3, 1, 2, 1, 2).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_squeeze', onnx_model, data, expected)


def test_tf_stack(runner):
    def my_func_1(x):
        return tf.stack([x[0], x[1], x[2]], axis=1)

    def my_func_2(x):
        return tf.stack([x[0], x[1], x[2]], axis=-1)

    for myFunc in [my_func_1, my_func_2]:
        K.clear_session()
        input_shape = (2, 3)
        input1 = Input(shape=input_shape)
        input2 = Input(shape=input_shape)
        input3 = Input(shape=input_shape)
        added = Lambda(myFunc)([input1, input2, input3])
        model = keras.models.Model(inputs=[input1, input2, input3], outputs=added)

        onnx_model = convert_keras(model, 'test_tf_stack')
        batch_data_shape = (1,) + input_shape
        data1 = np.random.rand(*batch_data_shape).astype(np.float32)
        data2 = np.random.rand(*batch_data_shape).astype(np.float32)
        data3 = np.random.rand(*batch_data_shape).astype(np.float32)
        expected = model.predict([data1, data2, data3])
        assert runner('onnx_stack', onnx_model, [data1, data2, data3], expected)


def test_stridedslice_with_version(runner):
    target_opset = get_maximum_opset_supported()
    for v1 in [-1, 1]:
        for v2 in [-1, 2]:
            model = Sequential()
            model.add(
                Lambda(lambda x: x[:, tf.newaxis, v1:, tf.newaxis, :v2, tf.newaxis, 3], input_shape=[2, 3, 4, 5]))
            onnx_model = convert_keras(model, 'test', target_opset=target_opset)

            data = np.random.rand(6 * 2 * 3 * 4 * 5).astype(np.float32).reshape(6, 2, 3, 4, 5)
            expected = model.predict(data)
            assert runner('onnx_stridedslice', onnx_model, data, expected)


def test_stridedslice_ellipse_newaxis(runner):
    target_opset = get_maximum_opset_supported()
    model = Sequential()
    model.add(
        Lambda(lambda x: x[:, 1:, tf.newaxis, ..., :, 1:, tf.newaxis], input_shape=[2, 3, 4, 3, 2, 2]))
    onnx_model = convert_keras(model, 'test', target_opset=target_opset)
    data = np.random.rand(6 * 2 * 3 * 4 * 3 * 2 * 2).astype(np.float32).reshape(6, 2, 3, 4, 3, 2, 2)
    expected = model.predict(data)
    assert runner('onnx_stridedslice', onnx_model, data, expected)

    model = Sequential()
    model.add(
        Lambda(lambda x: x[...], input_shape=[2, 3, 4, 5]))
    onnx_model = convert_keras(model, 'test', target_opset=target_opset)
    data = np.random.rand(6 * 2 * 3 * 4 * 5).astype(np.float32).reshape(6, 2, 3, 4, 5)
    expected = model.predict(data)
    assert runner('onnx_stridedslice', onnx_model, data, expected)


def test_stridedslice_ellipsis_mask_with_version(runner):
    target_opset = get_maximum_opset_supported()
    model = Sequential()
    model.add(Lambda(lambda x: x[:, :2, ..., 1:], input_shape=[3, 4, 5, 6, 3]))
    onnx_model = convert_keras(model, 'test', target_opset=target_opset)

    data = np.random.rand(5 * 3 * 4 * 5 * 6 * 3).astype(np.float32).reshape(5, 3, 4, 5, 6, 3)
    expected = model.predict(data)
    assert runner('onnx_stridedslice_ellipsis_mask', onnx_model, data, expected)


def test_stridedslice_shrink_mask_with_version(runner):
    target_opset = get_maximum_opset_supported()
    for shrink_value in [-1, 2]:
        model = Sequential()
        model.add(Lambda(lambda x: x[:, shrink_value, :], input_shape=[3, 4, 5]))
        onnx_model = convert_keras(model, 'test', target_opset=target_opset)
        data = np.random.rand(2 * 3 * 4 * 5).astype(np.float32).reshape(2, 3, 4, 5)
        expected = model.predict(data)
        assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 10,
                    reason="dynamic end is not supported for Slice op, opset < 10.")
def test_stridedslice_dynamic_end(runner):
    def my_func(x):
        frame_dim = tf.shape(x)[2]
        return x[:, :-1, 1:frame_dim - 1, :]

    model = Sequential()
    filters = 8
    kernel_size = (2, 5)
    strides = (1, 2)
    model.add(Conv2DTranspose(filters, kernel_size, strides=strides, use_bias=False,
                              padding="valid", name='conv2d_transpose', input_shape=[3, 4, 5]))
    model.add(Lambda(my_func))
    data1 = np.random.rand(2 * 3 * 4 * 5).astype(np.float32).reshape(2, 3, 4, 5)
    expected = model.predict(data1)
    onnx_model = convert_keras(model, 'test_strided_slice_dynamic_input')
    assert runner(onnx_model.graph.name, onnx_model, data1, expected)


@pytest.mark.skip("ConvTranspose does not support dynamic input for padding=same")
def test_conv_transpose_dynamic(runner):
    def my_func(x):
        frame_dim = tf.shape(x)[2]
        return x[:, :-1, 1:frame_dim - 1, :]

    model = Sequential()
    filters = 8
    kernel_size = (2, 5)
    strides = (1, 1)
    model.add(Lambda(my_func, input_shape=[3, 4, 5]))
    model.add(Conv2DTranspose(filters, kernel_size, strides=strides, use_bias=False,
                              padding="same", name='conv2d_transpose'))
    data1 = np.random.rand(2 * 3 * 4 * 5).astype(np.float32).reshape(2, 3, 4, 5)
    expected = model.predict(data1)
    onnx_model = convert_keras(model, 'test_conv_transpose_dynamic')
    assert runner(onnx_model.graph.name, onnx_model, data1, expected)


def test_tf_tile(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.tile(x, [1, 1, 3]), input_shape=[2, 2]))
    onnx_model = convert_keras(model, 'test_tf_tile')
    data = np.random.rand(3, 2, 2).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_tile', onnx_model, data, expected)


def test_tf_topk(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.nn.top_k(x, k=2)[0], input_shape=[5, 5]))
    onnx_model = convert_keras(model, 'test_tf_topk')
    data = np.random.rand(3, 5, 5).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_topk', onnx_model, data, expected)


def test_tf_transpose(runner):
    model = Sequential()
    model.add(Lambda(lambda x: tf.transpose(x, perm=[0, 2, 3, 1]), input_shape=[2, 3, 4]))
    onnx_model = convert_keras(model, 'test_tf_transpose')
    data = np.random.rand(2, 2, 3, 4).astype(np.float32)
    expected = model.predict(data)
    assert runner('onnx_transpose_1', onnx_model, data, expected)

    if is_tensorflow_later_than('1.13.0'):
        model = Sequential()
        model.add(Lambda(lambda x: tf.transpose(x), input_shape=[2, 3, 4]))
        onnx_model = convert_keras(model, 'test_tf_transpose')
        data = np.random.rand(4, 2, 3, 4).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_transpose_2', onnx_model, data, expected)

        def my_func_1(x):
            a = tf.constant([[1, 2, 3], [4, 5, 6]], tf.float32)
            return x + tf.transpose(a)

        model = Sequential()
        model.add(Lambda(lambda x: my_func_1(x), input_shape=[3, 2]))
        onnx_model = convert_keras(model, 'test_tf_transpose')
        data = np.random.rand(2, 3, 2).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_transpose_3', onnx_model, data, expected)


def test_tf_unpack(runner):
    for axis in [1, -1]:
        model = Sequential()
        model.add(Lambda(lambda x: tf.unstack(x, axis=axis)[0], input_shape=[2, 3, 4]))
        onnx_model = convert_keras(model, 'test_tf_unpack')
        data = np.random.rand(3, 2, 3, 4).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_unpack', onnx_model, data, expected)


@pytest.mark.skipif(is_tf2,
                    reason="tf 2.0 is not supported.")
def test_tf_variable(runner):
    val = np.random.random((2, 3, 4))
    for var_ in [K.variable(value=val), K.zeros(shape=(2, 3, 4)), K.ones(shape=(2, 3, 4))]:
        model = Sequential()
        model.add(Lambda(lambda x: x + var_, input_shape=[2, 3, 4]))
        onnx_model = convert_keras(model, 'test_tf_variable')
        data = np.random.rand(3, 2, 3, 4).astype(np.float32)
        expected = model.predict(data)
        assert runner('onnx_variable', onnx_model, data, expected)


@pytest.mark.skipif(is_tf2 or get_maximum_opset_supported() < 9,
                    reason="tf 2.0 or opset < 9 is not supported.")
def test_tf_where(runner):
    model = Sequential()
    a = tf.constant([[[1, 1], [3, 6]], [[7, 8], [9, 9]]])
    b = tf.where(tf.equal(a, 3))
    model.add(Lambda(lambda x: b, input_shape=(2,)))
    data = np.random.rand(1, 2).astype(np.float32)
    expected = model.predict(data)
    onnx_model = convert_keras(model, 'test_tf_where')
    assert runner('onnx_where', onnx_model, data, expected)

    model = Sequential()
    a = tf.constant([[[1, 1], [3, 6]], [[7, 8], [3, 3]]])
    b = tf.where(tf.equal(a, 3))
    model.add(Lambda(lambda x: b, input_shape=(2,)))
    data = np.random.rand(3, 2).astype(np.float32)
    expected = model.predict(data)
    onnx_model = convert_keras(model, 'test_tf_where')
    assert runner('onnx_where', onnx_model, data, expected)

    target_opset = get_maximum_opset_supported()
    if target_opset >= 9:
        model = Sequential()
        x = tf.constant([[1, 2, 3], [4, 5, 6]])
        y = tf.constant([[7, 8, 9], [10, 11, 12]])
        condition = tf.constant([[True, False, False], [False, True, True]])
        b = tf.where(condition, x, y)
        model.add(Lambda(lambda x: b, input_shape=(2,)))
        data = np.random.rand(2, 2).astype(np.float32)
        expected = model.predict(data)
        onnx_model = convert_keras(model, 'test_tf_where')
        assert runner('onnx_where', onnx_model, data, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 9, reason="conversion needs opset 9.")
def test_any_all(runner):
    for l_ in [keras.backend.any, keras.backend.all]:
        for axis in [1, -1]:
            keras_model = Sequential()
            keras_model.add(Lambda(lambda x: l_(x, axis=axis), input_shape=[3, 5]))
            onnx_model = convert_keras(keras_model, keras_model.name)
            x = np.random.rand(2, 3, 5).astype(np.float32)
            expected = keras_model.predict(x)
            assert runner(onnx_model.graph.name, onnx_model, x, expected)


def test_dense(runner):
    for bias_value in [True, False]:
        model = keras.Sequential()
        model.add(Dense(5, input_shape=(4,), activation='sigmoid'))
        model.add(Dense(3, input_shape=(5,), use_bias=bias_value))
        model.compile('sgd', 'mse')
        onnx_model = convert_keras(model, model.name)

        data = _asarray(1, 0, 0, 1)
        expected = model.predict(data)
        assert runner('dense', onnx_model, data, expected)


def test_dense_add(runner):
    input1 = Input(shape=(4,))
    x1 = Dense(3, activation='relu')(input1)
    input2 = Input(shape=(5,))
    x2 = Dense(3, activation='sigmoid')(input2)
    input3 = Input(shape=(3,))
    x3 = Dense(3)(input3)
    added = Add()([x1, x2, x3])  # equivalent to added = add([x1, x2])
    model = keras.models.Model(inputs=[input1, input2, input3], outputs=added)
    model.compile('sgd', 'mse')
    onnx_model = convert_keras(model, model.name)

    data = [_asarray(1.2, 2.4, -2, 1), _asarray(-1, -2, 0, 1, 2), _asarray(0.5, 1.5, -3.14159)]
    expected = model.predict(data)
    assert runner('onnx_dense_add', onnx_model, data, expected)


@pytest.mark.skipif(is_tf2, reason="const is not initialized this way for tf2")
def test_conv_add(runner):
    input1 = Input(shape=(10, 10, 1))
    x1 = Conv2D(32, strides=(2, 2), kernel_size=3,
                bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(input1)
    input2 = Input(tensor=tf.constant(np.random.rand(1, 32).astype(np.float32)))
    added = Add()([x1, input2])
    model = keras.models.Model(inputs=[input1, input2], outputs=added)
    onnx_model = convert_keras(model, model.name)
    data = [np.random.rand(1, 10, 10, 1).astype(np.float32)]
    expected = model.predict(data)
    data += [np.random.rand(1, 32).astype(np.float32)]
    assert runner('onnx_conv_add', onnx_model, data, expected)


def test_dense_softmax(runner):
    data = _asarray(1, 2, 3, 4)
    model = Sequential()
    model.add(Dense(5, input_shape=(4,), activation='softmax'))
    model.add(Dense(3, input_shape=(5,), use_bias=True))
    model.compile('sgd', 'mse')
    onnx_model = convert_keras(model, model.name)

    expected = model.predict(data)
    assert runner('dense_softmax_1', onnx_model, data, expected)

    model = Sequential()
    model.add(Dense(5, input_shape=(4,)))
    model.add(Activation('softmax'))
    model.add(Dense(3, input_shape=(5,), use_bias=True))
    model.compile('sgd', 'mse')
    onnx_model = convert_keras(model, model.name)

    expected = model.predict(data)
    assert runner('dense_softmax_2', onnx_model, data, expected)


@pytest.mark.parametrize("layer_type, data", [
    (Add, ([1, 2, 3], [4, 5, 6])),
    (Add, ([1, 2, 3], [4, 5, 6], [-3, -1, 1.5])),
    (Subtract, ([1, 2, 3], [4, 5, 6])),
    (Multiply, ([1, 2, 3], [4, 5, 6])),
    (Average, ([1, -2, 3], [3, 1, 1])),
    (Maximum, ([1, -2, 3], [3, 1, 1])),
    (lambda: Concatenate(), ([1, 2, 3], [4, 5, 6, 7])),
    (lambda: Concatenate(), ([1, 2, 3], [4, 5, 6, 7])),
    (lambda: Concatenate(-1), ([[1, 2], [3, 4]], [[4, 5], [6, 7]])),
    (lambda: Concatenate(1), ([[1, 2], [3, 4]], [[4, 5], [6, 7]])),
    (lambda: Concatenate(2), ([[1, 2], [3, 4]], [[4, 5], [6, 7]])),
])
def test_merge_layer(runner, layer_type, data):
    data2 = [_asarray(*d) for d in data]
    inputs = [Input(shape=d.shape[1:]) for d in data2]
    layer = layer_type()(inputs)
    model = keras.models.Model(inputs=inputs, outputs=layer)
    onnx_model = convert_keras(model, model.name)

    expected = model.predict(data2)
    assert runner(onnx_model.graph.name, onnx_model, data2, expected)


@pytest.fixture(scope='function')
def conv_runner(runner):
    def func(layer_type, input_channels, output_channels, kernel_size, strides, input_size, activation,
             rtol, atol, bias, channels_first=False, padding='valid'):
        model = keras.Sequential()
        input_size_seq = (input_size,) if isinstance(input_size, int) else input_size
        kwargs = {}
        if channels_first:
            input_shape = (input_channels,) + input_size_seq
            if not isinstance(layer_type, Conv1D):
                kwargs['data_format'] = 'channels_first'
        else:
            input_shape = input_size_seq + (input_channels,)

        model.add(layer_type(output_channels, kernel_size, input_shape=input_shape, strides=strides, padding=padding,
                             dilation_rate=1, activation=activation, use_bias=bias, **kwargs))
        data = np.random.uniform(-0.5, 0.5, size=(1,) + input_shape).astype(np.float32)
        onnx_model = convert_keras(model, model.name)

        expected = model.predict(data)
        assert runner(onnx_model.graph.name, onnx_model, data, expected, rtol=rtol, atol=atol)

    return func


@pytest.fixture(scope='function')
def conv1_runner(conv_runner):
    def func(*args, activation=None, rtol=1e-4, atol=1e-6, bias=False, padding='valid'):
        return conv_runner(Conv1D, *args, activation, rtol, atol, bias, padding=padding)

    return func


def test_conv1d(conv1_runner):
    conv1_runner(4, 5, 3, 1, 15)
    conv1_runner(4, 5, 3, 2, 15)


def test_conv1d_padding(conv1_runner):
    conv1_runner(4, 5, 3, 1, 15, padding='same')

    test_causal = False
    if is_tf_keras:
        import tensorflow
        from distutils.version import StrictVersion
        if StrictVersion(tensorflow.__version__.split('-')[0]) >= StrictVersion('1.12.0'):
            test_causal = True
    else:
        test_causal = True

    if test_causal:
        conv1_runner(4, 5, 3, 1, 15, padding='causal')


def test_conv1d_activation(conv1_runner):
    conv1_runner(4, 5, 3, 1, 15, activation='sigmoid')


def test_conv1d_bias(conv1_runner):
    conv1_runner(4, 5, 3, 1, 15, bias=True)


@pytest.fixture(scope='function')
def conv2_runner(conv_runner):
    def func(*args, activation=None, rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        input_dims = args[-1]
        assert len(input_dims) == 2
        conv_runner(Conv2D, *args, activation, rtol, atol, bias, channels_first, padding)

    return func


@pytest.fixture(scope='function')
def conv2trans_runner(conv_runner):
    def func(*args, activation=None, rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        input_dims = args[-1]
        assert len(input_dims) == 2
        conv_runner(Conv2DTranspose, *args, activation, rtol, atol, bias, channels_first, padding)

    return func


def test_conv2d(conv2_runner):
    conv2_runner(3, 5, (2, 2), (1, 1), (5, 5))


def test_conv2d_transpose(conv2trans_runner):
    conv2trans_runner(3, 5, (2, 2), (1, 1), (5, 5))


@pytest.mark.parametrize("padding", ["same", "valid"])
def test_conv2d_transpose_2(runner, padding):
    size = 128
    input_img = Input((size, size, 3))
    x = Conv2DTranspose(256, (4, 4), strides=2, use_bias=False, padding=padding,
                        kernel_initializer='he_normal')(input_img)
    y = BatchNormalization()(x)
    model = Model(inputs=input_img, outputs=y)
    data = np.random.rand(1, size, size, 3).astype(np.float32)
    onnx_model = convert_keras(model, model.name)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


def test_conv2d_padding_same(conv2_runner):
    conv2_runner(3, 5, (2, 2), (1, 1), (5, 5), padding='same')
    conv2_runner(8, 16, (1, 1), (2, 2), (60, 60), padding='same')
    conv2_runner(1, 1, (3, 3), (2, 2), (6, 6), padding='same')
    conv2_runner(1, 1, (7, 7), (2, 2), (25, 25), padding='same')
    conv2_runner(1, 1, (5, 7), (3, 5), (25, 25), padding='same')


@pytest.mark.skipif(is_tf_keras, reason="Generic conv implementation only supports NHWC tensor format in tf_keras")
def test_conv2d_format(conv2_runner):
    conv2_runner(3, 5, (2, 2), (1, 1), (5, 5), channels_first=True)


def test_conv2d_activation(conv2_runner):
    conv2_runner(3, 5, (2, 2), (1, 1), (5, 5), activation='relu')
    conv2_runner(3, 5, (2, 2), (1, 1), (5, 5), activation='softmax')


def test_conv2d_bias(conv2_runner):
    conv2_runner(3, 5, (2, 2), (1, 1), (5, 5), bias=True)


def test_conv2d_larger(conv2_runner):
    conv2_runner(3, 5, (7, 9), 1, (30, 20))


def test_conv2d_uneven_stride(conv2_runner):
    conv2_runner(3, 5, (4, 4), (3, 2), (20, 10))


@pytest.fixture(scope='function')
def conv3_runner(conv_runner):
    def func(*args, activation=None, rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        input_dims = args[-1]
        assert len(input_dims) == 3
        conv_runner(Conv3D, *args, activation, rtol, atol, bias, channels_first, padding)

    return func


def test_conv3d(conv3_runner):
    conv3_runner(3, 5, (2, 2, 2), (1, 1, 1), (5, 5, 8))


@pytest.fixture(scope='function')
def conv3trans_runner(conv_runner):
    def func(*args, activation=None, rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        input_dims = args[-1]
        assert len(input_dims) == 3
        conv_runner(Conv3DTranspose, *args, activation, rtol, atol, bias, channels_first, padding)

    return func


@pytest.mark.skip("ONNXRuntime doesn't support 3D ConvTranspose.")
def test_conv3d_transpose(conv3trans_runner):
    conv3trans_runner(3, 5, (2, 2, 2), (1, 1, 1), (5, 5, 8))


def test_flatten(runner):
    model = keras.Sequential()
    model.add(layers_core.Flatten(input_shape=(3, 2)))
    model.add(Dense(3))
    onnx_model = convert_keras(model, model.name)

    data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)
    expected = model.predict(data)
    assert runner('flatten', onnx_model, data, expected)


def test_opt_push_transpose_unsqueeze(runner):
    for input_shape_ in [(1, 5, 7), (1, 5)]:
        model = keras.Sequential()
        if len(input_shape_) == 3:
            model.add(Conv2D(64, (3, 3),
                             input_shape=input_shape_, padding='same', ))
        else:
            model.add(Conv1D(64, 3,
                             input_shape=input_shape_, padding='same', ))
        model.add(Lambda(lambda x: tf.squeeze(x, [1])))
        model.add(Lambda(lambda x: tf.expand_dims(x, 1)))
        onnx_model = convert_keras(model, model.name)
        batch_input_shape = (4,) + input_shape_
        x = np.random.rand(*batch_input_shape).astype(np.float32)
        expected = model.predict(x)
        assert runner(onnx_model.graph.name, onnx_model, x, expected)


def test_flatten2(runner):
    C = 3
    H = 5
    W = 7
    for data_format in ['channels_first', 'channels_last']:
        model = keras.Sequential()
        model.add(Conv2D(64, (3, 3),
                         input_shape=(C, H, W), padding='same', ))
        model.add(Flatten(data_format=data_format))
        onnx_model = convert_keras(model, model.name)
        x = np.random.rand(4, C, H, W).astype(np.float32)
        expected = model.predict(x)
        assert runner(onnx_model.graph.name, onnx_model, x, expected)


def test_reshape(runner):
    model = keras.Sequential()
    model.add(layers_core.Reshape((2, 3), input_shape=(3, 2)))
    onnx_model = convert_keras(model, model.name)

    data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)

    expected = model.predict(data)
    assert runner('reshape', onnx_model, data, expected)


def test_permute(runner):
    model = keras.Sequential()
    model.add(layers_core.Permute((2, 1), input_shape=(3, 2)))
    onnx_model = convert_keras(model, model.name)

    data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)

    expected = model.predict(data)
    assert runner('permute', onnx_model, data, expected)


def test_repeat_vector(runner):
    model = keras.Sequential()
    model.add(layers_core.RepeatVector(3, input_shape=(4,)))
    onnx_model = convert_keras(model, model.name)

    data = _asarray(1, 2, 3, 4)

    expected = model.predict(data)
    assert runner('repeat_vector', onnx_model, data, expected)


@pytest.fixture(scope='function')
def pooling_runner(runner):
    def func(layer, ishape, data_format='channels_last'):
        model = keras.Sequential()
        if is_keras_later_than('2.1.6'):
            nlayer = layer(data_format=data_format, input_shape=ishape) if \
                (layer.__name__.startswith("Global")) else layer(2, data_format=data_format, input_shape=ishape)
        else:
            nlayer = layer(input_shape=ishape) if \
                (layer.__name__.startswith("Global")) else layer(2, input_shape=ishape)

        model.add(nlayer)
        onnx_model = convert_keras(model, model.name)

        data = np.random.uniform(-0.5, 0.5, size=(1,) + ishape).astype(np.float32)

        expected = model.predict(data)
        assert runner(onnx_model.graph.name, onnx_model, data, expected)

    return func


def test_pooling_1d(pooling_runner):
    pooling_runner(AveragePooling1D, (4, 6))
    pooling_runner(MaxPool1D, (4, 6))
    if is_keras_later_than('2.1.6'):
        pooling_runner(AveragePooling1D, (4, 6), 'channels_first')
        pooling_runner(MaxPool1D, (4, 6), 'channels_first')


def test_pooling_2d(pooling_runner, runner):
    pooling_runner(AveragePooling2D, (4, 4, 3))

    N, C, H, W = 2, 3, 5, 5
    x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

    model = Sequential()
    model.add(MaxPooling2D((2, 2), strides=(2, 2), input_shape=(H, W, C), data_format='channels_last'))
    model.compile(optimizer='sgd', loss='mse')
    onnx_model = convert_keras(model, model.name)
    expected = model.predict(x)
    assert runner('max_pooling_2d', onnx_model, x, expected)

    # test padding='same'
    model = Sequential()
    model.add(
        MaxPooling2D((2, 2), strides=(2, 2), padding='same', input_shape=(H, W, C), data_format='channels_last'))
    model.compile(optimizer='sgd', loss='mse')
    onnx_model = convert_keras(model, model.name)
    expected = model.predict(x)
    assert runner('max_pooling_2d', onnx_model, x, expected)


def test_pooling_3d(pooling_runner):
    pooling_runner(AveragePooling3D, (4, 4, 4, 3))
    pooling_runner(MaxPool3D, (4, 4, 4, 3))


def test_pooling_global(pooling_runner):
    pooling_runner(GlobalAveragePooling2D, (4, 6, 2))


@pytest.mark.parametrize("layer", [
    'tanh',
    keras.activations.tanh,
    'sigmoid',
    keras.activations.sigmoid,
    'hard_sigmoid',
    keras.activations.hard_sigmoid,
    'relu',
    keras.activations.relu,
    'elu',
    keras.activations.elu,
    'selu',
    keras.activations.selu,
    'softsign',
    keras.activations.softsign,
    'softplus',
    keras.activations.softplus,
    'softmax',
    keras.activations.softmax,
    'linear',
    keras.activations.linear,
])
def test_activation_layer(runner, layer):
    data = _asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    layer = Activation(layer, input_shape=(data.size,))

    model = keras.Sequential()
    model.add(layer)
    onnx_model = convert_keras(model, model.name)

    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.fixture(scope='function')
def advanced_activation_runner(runner):
    def runner_func(layer, data, op_version=None):
        if op_version is None:
            op_version = get_maximum_opset_supported()

        model = keras.Sequential()
        model.add(layer)
        onnx_model = convert_keras(model, model.name, target_opset=op_version)

        expected = model.predict(data)
        assert runner(onnx_model.graph.name, onnx_model, data, expected)

    return runner_func


def test_selu(runner):
    SIZE = 10
    NB_CLASS = 5
    model = Sequential()
    model.add(Conv2D(32, strides=(2, 2), kernel_size=3, input_shape=(SIZE, SIZE, 1)))
    model.add(Flatten())
    model.add(Dense(32, activation='selu'))
    model.add(Dense(NB_CLASS, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    data = np.random.rand(5, SIZE, SIZE, 1).astype(np.float32)
    onnx_model = convert_keras(model, model.name)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


def test_gelu(runner):
    def gelu(x):
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    timesteps = 20
    a_dim = 10
    dense_hidden_size = 10
    dense_layer = Dense(dense_hidden_size, name="dense", trainable=False)

    mi = Input(shape=(timesteps, a_dim), dtype='float32', name="input")
    x1 = keras.layers.GlobalMaxPooling1D()(mi)
    x1 = dense_layer(x1)
    x1 = Activation(gelu, name='dense_gelu')(x1)
    keras_model = Model(inputs=mi, outputs=x1)
    data = np.random.rand(2, timesteps, a_dim).astype(np.float32)
    onnx_model = convert_keras(keras_model, keras_model.name)
    expected = keras_model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


def test_LeakyReLU(advanced_activation_runner):
    data = _asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    layer = advanced_activations.LeakyReLU(alpha=0.1, input_shape=(data.size,))
    advanced_activation_runner(layer, data)


@pytest.mark.skipif(get_maximum_opset_supported() < 8,
                    reason="ThresoldRelu needs ONNX opset 8")
def test_ThresholdedReLU(advanced_activation_runner):
    data = _asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    layer = advanced_activations.ThresholdedReLU(theta=1.0, input_shape=(data.size,))
    advanced_activation_runner(layer, data, op_version=8)

    layer = advanced_activations.ThresholdedReLU(theta=1.0, input_shape=(data.size,))
    advanced_activation_runner(layer, data)


def test_ELU(advanced_activation_runner):
    data = _asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    layer = advanced_activations.ELU(alpha=1.0, input_shape=(data.size,))
    advanced_activation_runner(layer, data)


def test_PReLU(advanced_activation_runner):
    data = _asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    layer = advanced_activations.PReLU(alpha_initializer='zeros', input_shape=(data.size,))
    advanced_activation_runner(layer, data)
    layer = advanced_activations.PReLU(alpha_initializer='ones', input_shape=(data.size,))
    advanced_activation_runner(layer, data)
    layer = advanced_activations.PReLU(alpha_initializer='RandomNormal', input_shape=(data.size,))
    advanced_activation_runner(layer, data)


@pytest.mark.skipif(is_keras_older_than("2.2.4"),
                    reason="ReLU needs keras 2.2.4+")
def test_ReLU(advanced_activation_runner):
    data = _asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    layer = ReLU(input_shape=(data.size,))
    advanced_activation_runner(layer, data)
    layer = ReLU(max_value=1.5, input_shape=(data.size,))
    advanced_activation_runner(layer, data)
    layer = ReLU(max_value=1.5, negative_slope=0.1, input_shape=(data.size,))
    advanced_activation_runner(layer, data)


def test_Softmax(advanced_activation_runner):
    data = _asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    layer = advanced_activations.Softmax(axis=-1, input_shape=(data.size,))
    advanced_activation_runner(layer, data)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_Softmax_2(runner, axis):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer((2, 3, 4)))
    model.add(keras.layers.Softmax(axis=axis))
    data = np.random.rand(2, 2, 3, 4).astype(np.float32)
    onnx_model = convert_keras(model, model.name)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.skipif(is_tensorflow_older_than('1.14.0') and is_tf_keras, reason='old tf version')
def test_tf_nn_activation(runner):
    for activation in ['relu', tf.nn.relu, tf.nn.relu6, tf.nn.softmax, tf.nn.leaky_relu]:
        model = keras.Sequential([
            Dense(64, activation=activation, input_shape=[10]),
            Dense(64, activation=activation),
            Dense(1)
        ])
        if is_tf_keras:
            model.add(Activation(tf.keras.layers.LeakyReLU(alpha=0.2)))
            model.add(Activation(tf.keras.layers.ReLU()))
            model.add(tf.keras.layers.PReLU())
            model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
            if is_tf2 and not is_tensorflow_older_than('2.2.0'):
                model.add(Lambda(lambda x: tf.keras.activations.swish(x)))
            if not is_tensorflow_older_than('1.15.0'):
                model.add(Lambda(lambda x: tf.nn.swish(x)))
        x = np.random.rand(5, 10).astype(np.float32)
        expected = model.predict(x)
        onnx_model = convert_keras(model, model.name)
        assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.fixture(scope='function')
def misc_conv_runner(runner):
    def func(layer, ishape, target_opset=None):
        if target_opset is None:
            target_opset = get_maximum_opset_supported()
        input = keras.Input(ishape)
        out = layer(input)
        model = keras.models.Model(input, out)
        onnx_model = convert_keras(model, model.name, target_opset=target_opset)

        data = np.random.uniform(0, 1, size=(1,) + ishape).astype(np.float32)

        expected = model.predict(data)
        assert runner(onnx_model.graph.name, onnx_model, data, expected)

    return func


def test_crop(misc_conv_runner):
    # It also passes the test for opset 9, we skip here because it uses a legacy experimental op DynamicSlice.
    opset_ = get_maximum_opset_supported()
    if opset_ >= 10:
        ishape = (10, 20)
        for crop_v in [2, (1, 2)]:
            layer = Cropping1D(cropping=crop_v)
            misc_conv_runner(layer, ishape, opset_)

        for data_format_ in ['channels_last', 'channels_first']:
            ishape = (20, 20, 10)
            for crop_v in [2, (2, 2), ((1, 2), (2, 3))]:
                layer = Cropping2D(cropping=crop_v, data_format=data_format_)
                misc_conv_runner(layer, ishape, opset_)
            ishape = (20, 20, 20, 10)
            for crop_v in [2, (2, 3, 4), ((1, 2), (2, 3), (3, 5))]:
                layer = Cropping3D(cropping=crop_v, data_format=data_format_)
                misc_conv_runner(layer, ishape, opset_)

    # TODO handle other cases for opset 8
    ishape = (20, 20, 1)
    layer = Cropping2D(cropping=((1, 2), (2, 3)), data_format='channels_last')
    misc_conv_runner(layer, ishape, opset_)


def test_upsample(misc_conv_runner):
    if is_keras_later_than('2.1.6'):
        ishape = (20, 5)
        layer = UpSampling1D(size=2)
        misc_conv_runner(layer, ishape)
        if not is_tf_keras:
            ishape = (20,)
            layer = UpSampling1D(size=2)
            misc_conv_runner(layer, ishape)
    ishape = (20, 20, 1)
    for size in [2, (2, 3)]:
        layer = UpSampling2D(size=size, data_format='channels_last')
        misc_conv_runner(layer, ishape)
        if not is_keras_older_than("2.2.3"):
            opset_ = get_maximum_opset_supported()
            if opset_ >= 11 or not is_tf_keras:
                layer = UpSampling2D(size=size, data_format='channels_last', interpolation='bilinear')
                misc_conv_runner(layer, ishape)
    ishape = (20, 20, 20, 1)
    layer = UpSampling3D(size=(2, 3, 4), data_format='channels_last')
    misc_conv_runner(layer, ishape)


def test_padding(misc_conv_runner):
    ishape = (20, 20, 1)
    layer = ZeroPadding2D(padding=((1, 2), (2, 3)), data_format='channels_last')
    misc_conv_runner(layer, ishape)


@pytest.mark.skipif(is_tf2 and is_tensorflow_older_than('2.2'),
                    reason="Variable freezing fails to replace ResourceGather op")
def test_embedding(runner):
    model = keras.Sequential()
    model.add(Embedding(1000, 64, input_length=10))
    input_array = np.random.randint(1000, size=(1, 10)).astype(np.float32)

    model.compile('rmsprop', 'mse')
    onnx_model = convert_keras(model, model.name)

    expected = model.predict(input_array)
    assert runner(onnx_model.graph.name, onnx_model, input_array, expected)


@pytest.fixture(scope='function')
def dot_runner(runner):
    def func(l2Normalize, input1, input2):
        data = [input1, input2]
        inputs = [Input(shape=d.shape[1:]) for d in data]

        layer = Dot(axes=-1, normalize=l2Normalize)(inputs)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        onnx_model = convert_keras(model, model.name)

        expected = model.predict(data)
        assert runner(onnx_model.graph.name, onnx_model, data, expected)

    return func


def test_dot(dot_runner):
    dot_runner(False, _asarray(1, 2, 3), _asarray(4, 5, 6))
    dot_runner(True, _asarray(1, 2, 3), _asarray(4, 5, 6))


def test_dot2(runner):
    input_1_shapes = [[32, 20, 1], [2, 3, 5], [2, 3, 5], [4, 3, 5], [2, 7], [2, 3, 4, 12, 3], [1, 3]]
    input_2_shapes = [[32, 30, 20], [2, 3, 5], [2, 3, 5], [4, 5], [2, 7, 5], [2, 3, 4, 15, 3], [1, 3]]
    axes_list = [[1, 2], 1, 2, [2, 1], [1, 1], 4, 1]
    for i_ in range(len(input_1_shapes)):
        for normalize in [True, False]:
            drop2_embed_title = Input(batch_shape=tuple(input_1_shapes[i_]), name='input1')
            att_weight = Input(batch_shape=tuple(input_2_shapes[i_]), name='input2')
            doc_vec1 = dot([drop2_embed_title, att_weight], axes=axes_list[i_], normalize=normalize)
            model = keras.models.Model(inputs=[drop2_embed_title, att_weight], outputs=doc_vec1)
            data1 = np.random.rand(*input_1_shapes[i_]).astype(np.float32)
            data2 = np.random.rand(*input_2_shapes[i_]).astype(np.float32)
            expected = model.predict([data1, data2])
            onnx_model = convert_keras(model, model.name)
            assert runner(onnx_model.graph.name, onnx_model, [data1, data2], expected)

    drop2_embed_title = Input(batch_shape=(None, 7), name='input1')
    att_weight = Input(batch_shape=(None, 7, 5), name='input2')
    doc_vec1 = dot([drop2_embed_title, att_weight], axes=[1, 1])
    model = keras.models.Model(inputs=[drop2_embed_title, att_weight], outputs=doc_vec1)
    data1 = np.random.rand(2, 7).astype(np.float32)
    data2 = np.random.rand(2, 7, 5).astype(np.float32)
    expected = model.predict([data1, data2])
    onnx_model = convert_keras(model, model.name)
    assert runner(onnx_model.graph.name, onnx_model, [data1, data2], expected)


def test_training_layer(runner):
    model = keras.Sequential()
    model.add(Dense(32, input_shape=(2, 3, 4)))
    model.add(GaussianNoise(0.1))
    model.add(Activation('relu'))
    model.add(GaussianDropout(0.1))
    model.add(AlphaDropout(0.1))
    model.add(SpatialDropout2D(0.2))
    model.add(Dense(1))
    onnx_model = convert_keras(model, model.name)
    data = np.random.rand(2, 2, 3, 4).astype(np.float32)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.fixture(scope='function')
def batch_norm_runner(runner):
    def func(data, gamma, beta, scale, center, axis):
        model = keras.Sequential()
        layer = BatchNormalization(
            axis=axis,
            input_shape=data.shape[1:],
            moving_mean_initializer=keras.initializers.constant(np.mean(data)),
            moving_variance_initializer=keras.initializers.constant(np.var(data)),
            gamma_initializer=gamma,
            beta_initializer=beta,
            center=center,
            scale=scale,
        )
        model.add(layer)
        onnx_model = convert_keras(model, model.name)

        expected = model.predict(data)
        assert runner(onnx_model.graph.name, onnx_model, data, expected)

    return func


def test_batch_normalization(batch_norm_runner):
    data = _asarray([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    batch_norm_runner(data, 'ones', 'zeros', True, True, 3)
    batch_norm_runner(data, 'ones', 'ones', True, True, 3)
    # The CPU implementation of FusedBatchNorm only supports NHWC tensor format in tf keras
    if not is_tf_keras:
        batch_norm_runner(data, 'ones', 'zeros', True, True, 1)
        batch_norm_runner(data, 'ones', 'ones', True, True, 1)
        batch_norm_runner(data, 'ones', 'ones', True, False, 1)
        batch_norm_runner(data, 'zeros', 'zeros', False, True, 1)


def test_batch_normalization_2(runner):
    # The CPU implementation of FusedBatchNorm only supports NHWC tensor format in tf keras
    axis_list = [-1] if is_tf_keras else [1, -1]
    for axis in axis_list:
        batch_size = 4
        input_dim_1 = 10
        input_dim_2 = 20
        input_dim_3 = 30

        model = Sequential()
        model.add(InputLayer(input_shape=(input_dim_1,)))
        model.add(BatchNormalization(axis=axis))
        model.add(Dense(5))
        data = np.random.randn(batch_size, input_dim_1).astype(np.float32)
        onnx_model = convert_keras(model)
        expected = model.predict(data)
        assert runner('test_batch_normalization_2_2d', onnx_model, [data], expected)

        model = Sequential()
        model.add(InputLayer(input_shape=(input_dim_1, input_dim_2)))
        if axis == -1:
            model.add(Conv1D(32, strides=(2,), kernel_size=3))
        model.add(BatchNormalization(axis=axis))
        model.add(Dense(5))
        data = np.random.randn(batch_size, input_dim_1, input_dim_2).astype(np.float32)
        onnx_model = convert_keras(model)
        expected = model.predict(data)
        assert runner('test_batch_normalization_2_3d', onnx_model, [data], expected)

        model = Sequential()
        model.add(InputLayer(input_shape=(input_dim_1, input_dim_2, input_dim_3)))
        if axis == -1:
            model.add(Conv2D(32, strides=(2, 2), kernel_size=3))
        model.add(BatchNormalization(axis=axis))
        model.add(Dense(5))
        data = np.random.randn(batch_size, input_dim_1, input_dim_2, input_dim_3).astype(np.float32)
        onnx_model = convert_keras(model)
        expected = model.predict(data)
        assert runner('test_batch_normalization_2_4d', onnx_model, [data], expected)


def test_simpleRNN(runner):
    K.clear_session()
    inputs1 = keras.Input(shape=(3, 1))
    cls = SimpleRNN(2, return_state=False, return_sequences=True)
    oname = cls(inputs1)  # , initial_state=t0)
    model = keras.Model(inputs=inputs1, outputs=[oname])
    onnx_model = convert_keras(model, model.name)

    data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)

    # with initial state
    inputs2 = keras.Input(shape=(1, 2))
    state = keras.Input(shape=(5,))
    hidden_1 = SimpleRNN(5, activation='relu', return_sequences=True)(inputs2, initial_state=[state])
    output = Dense(2, activation='sigmoid')(hidden_1)
    keras_model = keras.Model(inputs=[inputs2, state], outputs=output)
    onnx_model = convert_keras(keras_model, keras_model.name)

    N, H, W, C = 3, 1, 2, 5
    x = np.random.rand(N, H, W).astype(np.float32, copy=False)
    s = np.random.rand(N, C).astype(np.float32, copy=False)
    expected = keras_model.predict([x, s])
    assert runner(onnx_model.graph.name, onnx_model, [x, s], expected)

    # with initial state and output state
    input = keras.Input(shape=(1, 2))
    state_in = keras.Input(shape=(10,))
    hidden_1, state_out = SimpleRNN(10, activation='relu', return_sequences=True,
                                    return_state=True)(input, initial_state=[state_in])
    output = Dense(2, activation='linear')(hidden_1)
    keras_model = keras.Model(inputs=[input, state_in], outputs=[output, state_out])
    onnx_model = convert_keras(keras_model, keras_model.name)

    N, H, W, C = 3, 1, 2, 10
    x = np.random.rand(N, H, W).astype(np.float32, copy=False)
    s = np.random.rand(N, C).astype(np.float32, copy=False)
    expected = keras_model.predict([x, s])
    assert runner(onnx_model.graph.name, onnx_model, [x, s], expected)


@pytest.mark.parametrize("gru_class, rnn_version", GRU_CLASSES)
@pytest.mark.parametrize("return_sequences", [True, False])
def test_GRU(runner, gru_class, rnn_version, return_sequences):
    inputs1 = keras.Input(shape=(3, 1))

    # GRU with no initial state
    cls = gru_class(2, return_state=False, return_sequences=False)
    oname = cls(inputs1)
    model = keras.Model(inputs=inputs1, outputs=[oname])
    onnx_model = convert_keras(model, model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)

    # GRU with initial state
    cls = gru_class(2, return_state=False, return_sequences=return_sequences)
    initial_state_input = keras.Input(shape=(2,))
    oname = cls(inputs1, initial_state=initial_state_input)
    model = keras.Model(inputs=[inputs1, initial_state_input], outputs=[oname])
    onnx_model = convert_keras(model, model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
    init_state = np.array([0.4, 0.5]).astype(np.float32).reshape((1, 2))
    init_state_onnx = np.array([0.4, 0.5]).astype(np.float32).reshape((1, 2))
    expected = model.predict([data, init_state])
    assert runner(onnx_model.graph.name, onnx_model, [data, init_state_onnx], expected)


@pytest.mark.skipif(not is_tf_keras and is_tf2 and is_tensorflow_older_than('2.2'),
                    reason="Fails due to some reason involving bad graph captures. Works in new versions and tf_keras")
@pytest.mark.parametrize("gru_class, rnn_version", GRU_CLASSES)
def test_GRU_2(runner, gru_class, rnn_version):
    model = keras.Sequential(name='TestGRU')
    model.add(gru_class(400, reset_after=True, input_shape=(1, 257)))
    model.add(Dense(257, activation='sigmoid'))
    onnx_model = convert_keras(model, name=model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    data = np.random.rand(3, 257).astype(np.float32).reshape((3, 1, 257))
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
@pytest.mark.parametrize('return_sequences', [False, True])
@pytest.mark.parametrize('use_bias', [False, True])
def test_LSTM(runner, lstm_class, lstmcell_class, rnn_version, return_sequences, use_bias):
    inputs1 = keras.Input(shape=(3, 5))
    data = np.random.rand(3, 5).astype(np.float32).reshape((1, 3, 5))
    cls1 = lstm_class(units=2, return_state=True, return_sequences=return_sequences, use_bias=use_bias)
    cls2 = RNN(lstmcell_class(units=2, use_bias=use_bias), return_state=True, return_sequences=return_sequences)
    lstm1, state_h, state_c = cls1(inputs1)
    lstm2, state_h_2, state_c_2 = cls2(inputs1)
    model = keras.Model(inputs=inputs1, outputs=[lstm1, state_h, state_c, lstm2, state_h_2, state_c_2])
    onnx_model = convert_keras(model, model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.skipif((is_tensorflow_older_than('1.14.0') or (not is_tf_keras)), reason='old tf version')
@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
@pytest.mark.parametrize('return_sequences', [False, True])
@pytest.mark.parametrize('use_bias', [False, True])
def test_LSTM_rev(runner, lstm_class, lstmcell_class, rnn_version, return_sequences, use_bias):
    inputs1 = keras.Input(shape=(3, 5))
    data = np.random.rand(3, 5).astype(np.float32).reshape((1, 3, 5))
    cls = lstm_class(units=2, return_state=True, go_backwards=True, return_sequences=return_sequences, use_bias=use_bias)
    lstm1, state_h, state_c = cls(inputs1)
    model = keras.Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
    onnx_model = convert_keras(model, model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.skipif((is_tensorflow_older_than('1.14.0') or (not is_tf_keras)),
                    reason="keras LSTM does not have time_major attribute")
@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
def test_LSTM_time_major_return_seq_true(runner, lstm_class, lstmcell_class, rnn_version):
    inputs1 = keras.Input(shape=(3, 5))
    data = np.random.rand(1, 3, 5).astype(np.float32)
    # Transpose input to be time major
    input_transposed = tf.transpose(inputs1, perm=[1, 0, 2])
    lstm1, state_h, state_c = lstm_class(units=2, time_major=True, return_state=True,
                                   return_sequences=True)(input_transposed)
    lstm2, state_h_2, state_c_2 = RNN(lstmcell_class(units=2), time_major=True, return_state=True,
                                      return_sequences=True)(input_transposed)
    lstm1_trans = tf.transpose(lstm1, perm=[1, 0, 2])
    lstm2_trans = tf.transpose(lstm2, perm=[1,0,2])
    model = keras.Model(inputs=inputs1, outputs=[lstm1_trans, state_h, state_c,
                                                 lstm2_trans, state_h_2, state_c_2])
    onnx_model = convert_keras(model, model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.skipif((is_tensorflow_older_than('1.14.0') or (not is_tf_keras)),
                    reason="keras LSTM does not have time_major attribute")
@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
def test_LSTM_time_major_return_seq_false(runner, lstm_class, lstmcell_class, rnn_version):
    inputs1 = keras.Input(shape=(3, 5))
    data = np.random.rand(1, 3, 5).astype(np.float32)
    # Transpose input to be time major
    input_transposed = tf.transpose(inputs1, perm=[1, 0, 2])
    lstm1, state_h, state_c = lstm_class(units=2, time_major=True, return_state=True,
                                   return_sequences=False)(input_transposed)
    lstm2, state_h_2, state_c_2 = RNN(lstmcell_class(units=2), time_major=True, return_state=True,
                                  return_sequences=False)(input_transposed)
    model = keras.Model(inputs=inputs1, outputs=[lstm1, state_h, state_c,
                                                 lstm2, state_h_2, state_c_2])
    onnx_model = convert_keras(model, model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
def test_LSTM_with_bias(runner, lstm_class, lstmcell_class, rnn_version):
    inputs1 = keras.Input(shape=(1, 1))
    cls = lstm_class(units=1, return_state=True, return_sequences=True)
    lstm1, state_h, state_c = cls(inputs1)
    lstm2, state_h_2, state_c_2 = RNN(lstmcell_class(units=1), return_state=True,
                                      return_sequences=True)(inputs1)
    model = keras.Model(inputs=inputs1, outputs=[lstm1, state_h, state_c,
                                                 lstm2, state_h_2, state_c_2])
    # Set weights: kernel, recurrent_kernel and bias
    model.set_weights((np.array([[1, 2, 3, 4]]), np.array([[5, 6, 7, 8]]), np.array([1, 2, 3, 4]),
                       np.array([[1, 2, 3, 4]]), np.array([[5, 6, 7, 8]]), np.array([1, 2, 3, 4])))
    data = np.random.rand(1, 1).astype(np.float32).reshape((1, 1, 1))
    onnx_model = convert_keras(model, model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    expected = model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
def test_LSTM_reshape(runner, lstm_class, lstmcell_class, rnn_version):
    input_dim = 7
    sequence_len = 3
    inputs1 = keras.Input(shape=(sequence_len, input_dim))
    cls = lstm_class(units=5, return_state=False, return_sequences=True)
    lstm1 = cls(inputs1)
    lstm2 = RNN(lstmcell_class(units=5), return_state=False, return_sequences=True)(inputs1)

    output = Reshape((sequence_len, 5))(lstm1)
    output_2 = Reshape((sequence_len, 5))(lstm2)
    model = keras.Model(inputs=inputs1, outputs=[output, output_2])
    model.compile(optimizer='sgd', loss='mse')

    onnx_model = convert_keras(model, 'test')
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    data = np.random.rand(input_dim, sequence_len).astype(np.float32).reshape((1, sequence_len, input_dim))
    expected = model.predict(data)
    assert runner('tf_lstm', onnx_model, data, expected)


@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
def test_LSTM_with_initializer(runner, lstm_class, lstmcell_class, rnn_version):
    # batch_size = N
    # seq_length = H
    # input_size = W
    # hidden_size = C
    N, H, W, C = 3, 1, 2, 5

    # inputs shape: (batch_size, seq_length)
    inputs = keras.Input(shape=(H, W), name='inputs')

    # initial state shape: (hidden_size, 1)
    state_h = keras.Input(shape=(C,), name='state_h')
    state_c = keras.Input(shape=(C,), name='state_c')

    # create keras model
    lstm_layer = lstm_class(units=C, activation='relu', return_sequences=True)(inputs,
                                                                         initial_state=[state_h,
                                                                                        state_c])
    lstm_layer_2 = RNN(lstmcell_class(units=C, activation='relu'),
                       return_sequences=True)(inputs, initial_state=[state_h, state_c])
    outputs = Dense(W, activation='sigmoid')(lstm_layer)
    outputs_2 = Dense(W, activation='sigmoid')(lstm_layer_2)
    keras_model = keras.Model(inputs=[inputs, state_h, state_c],
                              outputs=[outputs, outputs_2])

    x = np.random.rand(1, H, W).astype(np.float32)
    sh = np.random.rand(1, C).astype(np.float32)
    sc = np.random.rand(1, C).astype(np.float32)
    onnx_model = convert_keras(keras_model, keras_model.name)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    expected = keras_model.predict([x, sh, sc])
    assert runner(onnx_model.graph.name, onnx_model, {"inputs": x, 'state_h': sh, 'state_c': sc}, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 5,
                    reason="None seq_length LSTM is not supported before opset 5.")
@pytest.mark.skipif(is_tensorflow_older_than('2.2'), reason='require 2.2 to fix freezing')
@pytest.mark.parametrize("lstm_class, lstmcell_class, rnn_version", LSTM_CLASSES)
@pytest.mark.parametrize('return_sequences', [False, True])
def test_LSTM_seqlen_none(runner, lstm_class, lstmcell_class, rnn_version, return_sequences):
    lstm_dim = 2
    data = np.random.rand(1, 5, 1).astype(np.float32)
    inp = Input(batch_shape=(1, None, 1))
    out = lstm_class(lstm_dim, return_sequences=return_sequences, stateful=True)(inp)
    out_2 = RNN(lstmcell_class(lstm_dim), return_sequences=return_sequences, stateful=True)(inp)
    keras_model = keras.Model(inputs=inp, outputs=[out, out_2])

    onnx_model = convert_keras(keras_model)
    if rnn_version == "v2":
        assert no_loops_in_tf2(onnx_model)
    expected = keras_model.predict(data)
    assert runner(onnx_model.graph.name, onnx_model, data, expected)


@pytest.mark.parametrize("return_sequences", [True, False])
@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_bidirectional(runner, rnn_class, return_sequences):
    input_dim = 10
    sequence_len = 5
    op_version = get_maximum_opset_supported()
    batch_list = [1, 4] if op_version >= 9 else [1]

    model = keras.Sequential()
    model.add(Bidirectional(rnn_class(7, return_sequences=return_sequences),
                            input_shape=(5, 10)))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    onnx_model = convert_keras(model, 'test', target_opset=op_version)
    assert all_recurrents_should_bidirectional(onnx_model)
    for batch in batch_list:
        data = np.random.rand(batch, sequence_len, input_dim).astype(np.float32)
        expected = model.predict(data)
        assert runner('bidirectional', onnx_model, data, expected)

    for merge_mode in ['concat', None]:
        sub_input1 = Input(shape=(sequence_len, input_dim))
        sub_mapped1 = Bidirectional(rnn_class(7, return_sequences=return_sequences),
                                    input_shape=(5, 10), merge_mode=merge_mode)(sub_input1)
        keras_model = keras.Model(inputs=sub_input1, outputs=sub_mapped1)
        onnx_model = convert_keras(keras_model, 'test_2', target_opset=op_version)
        assert all_recurrents_should_bidirectional(onnx_model)
        for batch in batch_list:
            data = np.random.rand(batch, sequence_len, input_dim).astype(np.float32)
            expected = keras_model.predict(data)
            assert runner('bidirectional', onnx_model, data, expected)


@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_bidirectional_with_bias(runner, rnn_class):
    K.clear_session()
    model = keras.Sequential()
    model.add(Bidirectional(rnn_class(4, return_sequences=False),
                            input_shape=(3, 5), name='bi'))

    x = np.random.uniform(100, 999, size=(2, 3, 5)).astype(np.float32)

    # Test with the default bias
    expected = model.predict(x)
    onnx_model = convert_keras(model, model.name)
    assert all_recurrents_should_bidirectional(onnx_model)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)

    # Set bias values to random floats
    rnn_layer = model.get_layer('bi')
    weights = rnn_layer.get_weights()
    weights[2] = np.random.uniform(size=weights[2].shape)
    weights[5] = weights[2]
    rnn_layer.set_weights(weights)

    # Test with random bias
    expected = model.predict(x)
    onnx_model = convert_keras(model, model.name)
    assert all_recurrents_should_bidirectional(onnx_model)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.mark.skipif((is_tensorflow_older_than('2.3.0') or (not is_tf_keras)),
                    reason=(
                    "keras LSTM does not have time_major attribute. There was a bug in tf.keras bidirectional lstm with time_major true which will be fixed in tf-2.3, See - https://github.com/tensorflow/tensorflow/issues/39635"))
@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_bidirectional_time_major_true(runner, rnn_class):
    feature_dim = 1
    seq_len = 3
    x = np.ones((1, seq_len, feature_dim), dtype=np.float32)

    for ret_seq in [True, False]:
        for merge_mode in ['concat', None]:
            K.clear_session()
            input = keras.Input(shape=(seq_len, feature_dim))
            # Transpose input to be time major
            input_transposed = tf.transpose(input, perm=[1, 0, 2])
            output = Bidirectional(rnn_class(1, return_sequences=ret_seq,
                                             time_major=True),
                                   name='bi', merge_mode=merge_mode)(input_transposed)
            if ret_seq and merge_mode == 'concat':
                output = tf.transpose(output, perm=[1, 0, 2])
            model = keras.Model(inputs=input, outputs=output)

            expected = model.predict(x)
            onnx_model = convert_keras(model, model.name)
            assert all_recurrents_should_bidirectional(onnx_model)
            assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_bidirectional_with_initial_states(runner, rnn_class):
    input1 = Input(shape=(None, 5))
    states = Bidirectional(rnn_class(2, return_state=True))(input1)
    model = Model(input1, states)

    x = np.random.uniform(0.1, 1.0, size=(4, 3, 5)).astype(np.float32)
    inputs = [x]

    expected = model.predict(inputs)
    onnx_model = convert_keras(model, model.name)
    assert all_recurrents_should_bidirectional(onnx_model)
    assert runner(onnx_model.graph.name, onnx_model, inputs, expected)

    input2 = Input(shape=(None, 5))
    states = Bidirectional(rnn_class(2, return_state=True))(input1)[1:]
    out = Bidirectional(rnn_class(2, return_sequences=True))(input2, initial_state=states)
    model = Model([input1, input2], out)
    inputs = [x, x]

    expected = model.predict(inputs)
    onnx_model = convert_keras(model, model.name)
    assert all_recurrents_should_bidirectional(onnx_model)
    assert runner(onnx_model.graph.name, onnx_model, inputs, expected, atol=1e-5)


@pytest.mark.skipif(get_maximum_opset_supported() < 5,
                    reason="None seq_length Bidirectional LSTM is not supported before opset 5.")
@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
@pytest.mark.skipif(is_tf2 and is_tensorflow_older_than('2.2'),
                    reason="Variable freezing fails to replace GatherResource op")
def test_bidirectional_seqlen_none(runner, rnn_class):
    model = Sequential()
    model.add(Embedding(39, 128))
    model.add(Bidirectional(rnn_class(256, input_shape=(None, 32), return_sequences=True)))
    model.add(Dense(44))

    onnx_model = convert_keras(model, model.name)
    assert all_recurrents_should_bidirectional(onnx_model)
    for batch in [1, 4]:
        x = np.random.rand(batch, 50).astype(np.float32)
        expected = model.predict(x)
        assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_rnn_state_passing(runner, rnn_class):
    input1 = Input(shape=(None, 5))
    input2 = Input(shape=(None, 5))

    states = rnn_class(2, return_state=True)(input1)[1:]
    out = rnn_class(2, return_sequences=True)(input2, initial_state=states)
    model = Model([input1, input2], out)

    x = np.random.uniform(0.1, 1.0, size=(4, 3, 5)).astype(np.float32)
    inputs = [x, x]

    expected = model.predict(inputs)
    onnx_model = convert_keras(model, model.name)
    assert runner(onnx_model.graph.name, onnx_model, inputs, expected, atol=1e-5)


def test_seq_dynamic_batch_size(runner):
    K.clear_session()
    data_dim = 4  # input_size
    timesteps = 3  # seq_length

    # expected input data shape: (batch_size, timesteps, data_dim)
    test_input = np.random.random_sample((100, timesteps, data_dim))
    test_output = np.random.random_sample((100, 128))

    # Number of layer and number of neurons in each layer
    num_neur = [128, 256, 128]
    epochs = 200
    batch_size = 50
    nodeFuncList = [SimpleRNN, GRU, LSTM]

    for nodeFunc in nodeFuncList:
        model = Sequential()
        for i in range(len(num_neur)):  # multi-layer
            if len(num_neur) == 1:
                model.add(nodeFunc(num_neur[i], input_shape=(timesteps, data_dim), unroll=True))
            else:
                if i < len(num_neur) - 1:
                    model.add(
                        nodeFunc(num_neur[i], input_shape=(timesteps, data_dim), return_sequences=True,
                                 unroll=True))
                else:
                    model.add(nodeFunc(num_neur[i], input_shape=(timesteps, data_dim), unroll=True))

        # Compile the neural network
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(test_input, test_output, epochs=epochs, batch_size=batch_size, verbose=0)
        test_input = np.random.random_sample((5, timesteps, data_dim)).astype(np.float32)
        test_output = model.predict(test_input)
        onnx_model = convert_keras(model, model.name)
        assert runner(onnx_model.graph.name, onnx_model, test_input, test_output)


def test_separable_convolution(runner):
    N, C, H, W = 2, 3, 5, 5
    x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)
    model = Sequential()
    model.add(
        SeparableConv2D(filters=10, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                        data_format='channels_last', depth_multiplier=4))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))
    model.compile(optimizer='sgd', loss='mse')
    onnx_model = convert_keras(model, 'test')
    expected = model.predict(x)
    assert runner('separable_convolution_1', onnx_model, x, expected)

    x = np.random.rand(N, H, C).astype(np.float32, copy=False)
    model = Sequential()
    model.add(SeparableConv1D(filters=10, kernel_size=2, strides=1, padding='valid', input_shape=(H, C),
                              data_format='channels_last'))
    model.compile(optimizer='sgd', loss='mse')
    onnx_model = convert_keras(model, 'test')
    expected = model.predict(x)
    assert runner('separable_convolution_2', onnx_model, x, expected)


@pytest.mark.skipif(is_tf2 and is_tensorflow_older_than('2.2'),
                    reason="Variable freezing fails to replace GatherResource op")
def test_shared_embed(runner):
    max_cont_length = 5
    max_ques_length = 7
    word_dict_len = 10
    word_dim = 6
    h_word_mat = 'aa'
    # Input Embedding Layer
    contw_input_ = Input((max_cont_length,))  # [bs, c_len]
    quesw_input_ = Input((max_ques_length,))  # [bs, q_len]

    # embedding word
    WordEmbedding = Embedding(word_dict_len, word_dim, trainable=False,
                              name="word_embedding_" + h_word_mat)
    xw_cont = Dropout(0.2)(WordEmbedding(contw_input_))  # [bs, c_len, word_dim]
    xw_ques = Dropout(0.2)(WordEmbedding(quesw_input_))  # [bs, c_len, word_dim]

    keras_model = keras.models.Model(inputs=[contw_input_, quesw_input_],
                                     outputs=[xw_cont, xw_ques])
    onnx_model = convert_keras(keras_model, keras_model.name)
    batch_size = 3
    x = np.random.rand(batch_size, max_cont_length).astype(np.float32)
    y = np.random.rand(batch_size, max_ques_length).astype(np.float32)
    expected = keras_model.predict([x, y])
    assert runner(onnx_model.graph.name, onnx_model, [x, y], expected)


def test_recursive_model(runner):
    keras.backend.set_learning_phase(0)

    N, C, D = 2, 3, 3
    x = np.random.rand(N, C).astype(np.float32, copy=False)

    sub_input1 = Input(shape=(C,))
    sub_mapped1 = Dense(D)(sub_input1)
    sub_model1 = keras.Model(inputs=sub_input1, outputs=sub_mapped1)

    sub_input2 = Input(shape=(C,))
    sub_mapped2 = Dense(D)(sub_input2)
    sub_model2 = keras.Model(inputs=sub_input2, outputs=sub_mapped2)

    input1 = Input(shape=(D,))
    input2 = Input(shape=(D,))
    mapped1_2 = sub_model1(input1)
    mapped2_2 = sub_model2(input2)
    sub_sum = Add()([mapped1_2, mapped2_2])
    keras_model = keras.Model(inputs=[input1, input2], outputs=sub_sum)
    onnx_model = convert_keras(keras_model, keras_model.name)

    x = [x, 2 * x]
    expected = keras_model.predict(x)
    assert runner('recursive', onnx_model, x, expected)


def test_recursive_and_shared_model(runner):
    keras.backend.set_learning_phase(0)
    N, C, D = 2, 3, 3
    x = np.random.rand(N, C).astype(np.float32, copy=False)

    sub_input1 = Input(shape=(C,))
    sub_mapped1 = Dense(D)(sub_input1)
    sub_output1 = Activation('sigmoid')(sub_mapped1)
    sub_model1 = keras.Model(inputs=sub_input1, outputs=sub_output1)

    sub_input2 = Input(shape=(C,))
    sub_mapped2 = sub_model1(sub_input2)
    sub_output2 = Activation('tanh')(sub_mapped2)
    sub_model2 = keras.Model(inputs=sub_input2, outputs=sub_output2)

    input1 = Input(shape=(D,))
    input2 = Input(shape=(D,))
    mapped1_1 = Activation('tanh')(input1)
    mapped2_1 = Activation('sigmoid')(input2)
    mapped1_2 = sub_model1(mapped1_1)
    mapped1_3 = sub_model1(mapped1_2)
    mapped2_2 = sub_model2(mapped2_1)
    sub_sum = Add()([mapped1_3, mapped2_2])
    keras_model = keras.Model(inputs=[input1, input2], outputs=sub_sum)
    keras_model.compile('sgd', loss='mse')
    onnx_model = convert_keras(keras_model, keras_model.name)

    x = [x, 2 * x]
    expected = keras_model.predict(x)
    assert runner('recursive_and_shared', onnx_model, x, expected)


@pytest.mark.skipif(is_keras_older_than("2.2.4"),
                    reason="Low keras version is not supported.")
def test_shared_model_2(runner):
    K.set_learning_phase(0)

    def _conv_layer(input, filters, kernel_size, relu_flag=False, strides=1, dilation_rate=1):
        padding = 'same' if strides == 1 else 'valid'
        if strides > 1:
            input = ZeroPadding2D(((0, 1), (0, 1)), data_format=K.image_data_format())(input)
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                   padding=padding, use_bias=False, dilation_rate=dilation_rate)(input)
        ch_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = BatchNormalization(axis=ch_axis)(x)
        if relu_flag:
            return ReLU()(x)
        else:
            return x

    def _model(relu_flag=False):
        input = Input(shape=(3, 320, 320), name='input_1')
        x = _conv_layer(input, 16, 3, relu_flag)
        return Model(inputs=input, outputs=x, name='backbone')

    relu_flags = [False] if is_tf2 or is_tf_keras else [True, False]
    for relu_flag_ in relu_flags:
        input = Input(shape=(3, 320, 320), name='input')
        backbone = _model(relu_flag_)
        x = backbone(input)
        x = _conv_layer(x, 16, 3)
        model = Model(inputs=[input], outputs=[x])

        onnx_model = convert_keras(model, model.name)
        x = np.random.rand(2, 3, 320, 320).astype(np.float32)
        expected = model.predict(x)
        assert runner(onnx_model.graph.name, onnx_model, x, expected, atol=1e-5)


@pytest.mark.skipif(is_keras_older_than("2.2.4"),
                    reason="ReLU support requires keras 2.2.4 or later.")
def test_shared_model_3(runner):
    def _bottleneck(x, filters, activation, strides, block_id):
        padding = 'same' if strides == 1 else 'valid'
        ch_axis = 1 if K.image_data_format() == 'channels_first' else -1
        if strides > 1:
            x = ZeroPadding2D(((0, 1), (0, 1)), data_format=K.image_data_format())(x)

        x = Conv2D(filters // 2, (1, 1), padding='same', name='bottleneck_' + str(block_id) + '_conv_0',
                   use_bias=False, data_format=K.image_data_format())(x)

        x = BatchNormalization(axis=ch_axis, name='bottleneck_' + str(block_id) + '_bnorm_0')(x)

        if activation == 'relu':
            x = ReLU(name='bottleneck_' + str(block_id) + '_relu_0')(x)
        elif activation == 'leaky':
            x = LeakyReLU(name='bottleneck_' + str(block_id) + '_leaky_0')(x)
        else:
            assert False

        x = Conv2D(filters // 2, (3, 3), padding=padding, name='bottleneck_' + str(block_id) + '_conv_1',
                   strides=strides, use_bias=False, data_format=K.image_data_format())(x)
        x = BatchNormalization(axis=ch_axis, name='bottleneck_' + str(block_id) + '_bnorm_1')(x)
        if activation == 'relu':
            x = ReLU(name='bottleneck_' + str(block_id) + '_relu_1')(x)
        elif activation == 'leaky':
            x = LeakyReLU(name='bottleneck_' + str(block_id) + '_leaky_1')(x)
        else:
            assert False

        x = Conv2D(filters, (1, 1), padding='same', name='bottleneck_' + str(block_id) + '_conv_2',
                   use_bias=False, data_format=K.image_data_format())(x)
        x = BatchNormalization(axis=ch_axis, name='bottleneck_' + str(block_id) + '_bnorm_2')(x)
        if activation == 'relu':
            x = ReLU(name='bottleneck_' + str(block_id) + '_relu_2')(x)
        elif activation == 'leaky':
            x = LeakyReLU(name='bottleneck_' + str(block_id) + '_leaky_2')(x)
        else:
            assert False

        return x

    def convnet_7(input_shape, activation):
        input = Input(shape=input_shape, name='input_1')
        x = _bottleneck(input, filters=16, strides=1, activation=activation, block_id=1)
        x = _bottleneck(x, filters=32, strides=2, activation=activation, block_id=2)
        return Model(inputs=input, outputs=x, name='convnet_7')

    activation_list = ['leaky'] if is_tf2 or is_tf_keras else ['relu', 'leaky']
    for activation in activation_list:
        model = convnet_7(input_shape=(3, 96, 128), activation=activation)
        onnx_model = convert_keras(model, model.name)
        x = np.random.rand(1, 3, 96, 128).astype(np.float32)
        expected = model.predict(x)
        assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_masking(runner, rnn_class):
    timesteps, features = (3, 5)
    model = Sequential([
        keras.layers.Masking(mask_value=0., input_shape=(timesteps, features)),
        rnn_class(8, return_state=False, return_sequences=False)
    ])

    onnx_model = convert_keras(model, model.name)
    x = np.random.uniform(100, 999, size=(2, 3, 5)).astype(np.float32)
    expected = model.predict(x)
    assert runner(onnx_model.graph.name, onnx_model, x, expected, rtol=5e-3)


@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_masking_bias(runner, rnn_class):
    timesteps, features = (3, 5)
    model = Sequential([
        keras.layers.Masking(mask_value=0., input_shape=(timesteps, features)),
        rnn_class(8, return_state=False, return_sequences=False, use_bias=True, name='rnn')
    ])

    x = np.random.uniform(100, 999, size=(2, 3, 5)).astype(np.float32)
    # Fill one of the entries with all zeros except the first timestep
    x[1, 1:, :] = 0

    # Test with the default bias
    expected = model.predict(x)
    onnx_model = convert_keras(model, model.name)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)

    # Set bias values to random floats
    rnn_layer = model.get_layer('rnn')
    weights = rnn_layer.get_weights()
    weights[2] = np.random.uniform(size=weights[2].shape)
    rnn_layer.set_weights(weights)

    # Test with random bias
    expected = model.predict(x)
    onnx_model = convert_keras(model, model.name)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 9, reason='bidirectional is not supported for opset < 9')
@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_masking_bias_bidirectional(runner, rnn_class):
    timesteps, features = (3, 5)
    model = Sequential([
        keras.layers.Masking(mask_value=0., input_shape=(timesteps, features)),
        Bidirectional(rnn_class(8, return_state=False, return_sequences=False, use_bias=True), name='bi')
    ])

    x = np.random.uniform(100, 999, size=(2, 3, 5)).astype(np.float32)
    # Fill one of the entries with all zeros except the first timestep
    x[1, 1:, :] = 0

    # Test with the default bias
    expected = model.predict(x)
    onnx_model = convert_keras(model, model.name)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)

    # Set bias values to random floats
    rnn_layer = model.get_layer('bi')
    weights = rnn_layer.get_weights()
    weights[2] = np.random.uniform(size=weights[2].shape)
    weights[5] = weights[2]
    rnn_layer.set_weights(weights)

    # Test with random bias
    expected = model.predict(x)
    onnx_model = convert_keras(model, model.name)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.mark.parametrize("rnn_class", RNN_CLASSES)
def test_masking_value(runner, rnn_class):
    if rnn_class is SimpleRNN:
        pytest.skip('SimpleRNN intermittently fails this test')

    timesteps, features = (3, 5)
    mask_value = 5.
    model = Sequential([
        keras.layers.Masking(mask_value=mask_value, input_shape=(timesteps, features)),
        rnn_class(8, return_state=False, return_sequences=False)
    ])

    onnx_model = convert_keras(model, model.name)
    x = np.random.uniform(100, 999, size=(2, 3, 5)).astype(np.float32)
    x[1, :, :] = mask_value
    expected = model.predict(x)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)


def test_masking_custom(runner):
    class MyPoolingMask(keras.layers.Layer):
        def __init__(self, **kwargs):
            self.supports_masking = True
            super(MyPoolingMask, self).__init__(**kwargs)

        def build(self, input_shape):
            super(MyPoolingMask, self).build(input_shape)

        def compute_mask(self, inputs, input_mask=None):
            return None

        def call(self, inputs, mask=None, **kwargs):
            if mask is not None:
                return K.sum(inputs, axis=-2) / (
                        K.sum(K.cast(mask, K.dtype(inputs)), axis=-1, keepdims=True) + K.epsilon())
            else:
                output = K.mean(inputs, axis=-2)
                return output

        def compute_output_shape(self, input_shape):
            return input_shape[:-2] + input_shape[-1:]

    timesteps, features = (3, 5)
    model = Sequential([
        keras.layers.Masking(mask_value=0., input_shape=(timesteps, features)),
        MyPoolingMask()
    ])

    onnx_model = convert_keras(model, model.name)
    x = np.random.uniform(100, 999, size=(2, 3, 5)).astype(np.float32)
    expected = model.predict(x)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)


def test_timedistributed(runner):
    keras_model = keras.Sequential()
    keras_model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
    # keras_model.output_shape == (None, 10, 8)
    onnx_model = convert_keras(keras_model, keras_model.name)
    x = np.random.rand(32, 10, 16).astype(np.float32)
    expected = keras_model.predict(x)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)

    keras_model = keras.Sequential()
    N, D, W, H, C = 5, 10, 15, 15, 3
    keras_model.add(TimeDistributed(Conv2D(64, (3, 3)),
                                    input_shape=(D, W, H, C)))
    onnx_model = convert_keras(keras_model, keras_model.name)
    x = np.random.rand(N, D, W, H, C).astype(np.float32)
    expected = keras_model.predict(x)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)


def test_channel_first_input(runner):
    N, W, H, C = 2, 5, 6, 3
    inp1 = Input(batch_shape=(N, W, H, C), name='input1')
    inp2 = Input(batch_shape=(N, W, H, C), name='input2')
    output = Add()([inp1, inp2])
    model = keras.models.Model(inputs=[inp1, inp2], outputs=output)
    onnx_model = convert_keras(model, model.name, channel_first_inputs=['input1'])
    assert onnx_model is not None

    data1 = np.random.rand(N, W, H, C).astype(np.float32).reshape((N, W, H, C))
    data2 = np.random.rand(N, W, H, C).astype(np.float32).reshape((N, W, H, C))
    data_transpose = np.transpose(data1, (0, 3, 1, 2))
    assert data_transpose.shape == (N, C, W, H)

    expected = model.predict([data1, data2])
    assert runner('channel_first_input', onnx_model, [data_transpose, data2], expected)


def test_channel_last(runner):
    N, C, H, W = 2, 3, 5, 5
    x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

    model = Sequential()
    model.add(Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                     data_format='channels_last'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))

    model.compile(optimizer='sgd', loss='mse')
    onnx_model = convert_keras(model, channel_first_inputs=[model.input_names[0]])

    expected = model.predict(x)
    assert expected is not None
    assert onnx_model is not None

    x = np.transpose(x.astype(np.float32), [0, 3, 1, 2])
    assert runner('channel_last_input', onnx_model, x, expected)


def test_sub_model(runner):
    class IdentityLayer(Layer):
        def __init__(self, **kwargs):
            super(IdentityLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            super(IdentityLayer, self).build(input_shape)

        def call(self, inputs, training=None):
            return inputs

        def compute_output_shape(self, input_shape):
            return input_shape

    input_shape = [700, 420, 1]
    num_classes = 10

    image_input = Input(shape=input_shape, name='image_input')

    model = Sequential()  # 28, 28, 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape, padding='valid'))  # 28, 28, 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))  # 28, 28, 1
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))  # 14, 14, 1
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(12, 12), strides=(14, 14), padding="valid", activation='relu'))
    model.add(Dropout(0.5))

    features = model(image_input)

    outputs = []
    for _ in range(3):
        output1 = Dense(num_classes, activation="softmax")(
            Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
        output2 = Dense(1, activation="sigmoid")(
            Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
        output3 = Dense(2, activation="tanh")(
            Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
        output4 = Dense(2, activation="tanh")(
            Dense(64, activation="relu")(Dense(128, activation="relu")(features)))
        outputs += [output1, output2, output3, output4]

    output = Concatenate(name="output")(outputs)
    output = IdentityLayer()(output)
    model1 = Model(image_input, output)
    onnx_model = convert_keras(model1, model1.name)
    x = np.random.rand(2, 700, 420, 1).astype(np.float32)
    expected = model1.predict(x)
    assert runner(onnx_model.graph.name, onnx_model, x, expected)


@pytest.mark.skipif((is_tensorflow_older_than('1.14.0') or (not is_tf_keras)), reason='old tf version')
def test_reverseV2(runner):
    input = Input(shape=(2, 4), name='input')
    rev = tf.reverse(input, [1])
    model = tf.keras.models.Model(inputs=input, outputs=rev)

    onnx_model = convert_keras(model, 'tf_rev_v2')
    data = np.random.rand(1, 2, 4).astype(np.float32)
    expected = model.predict(data)
    assert runner('tf_rev_v2', onnx_model, data, expected)


@pytest.mark.skipif(get_maximum_opset_supported() < 11, reason="TensorScatterUpdate is not supported before opset 11.")
@pytest.mark.skipif((is_tensorflow_older_than('1.14.0') or (not is_tf_keras)), reason='old tf version')
def test_tensor_scatter_update(runner):
    indices = np.array([[0],[0]]).astype(np.int64)
    updates = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).astype(np.float32)
    tensor = Input(shape=(2,2), name='tensor')

    out = tf.tensor_scatter_nd_update(tensor, indices, updates)
    model = tf.keras.models.Model(inputs=tensor, outputs=out)
    onnx_model = convert_keras(model, model.name)

    tensor_data = np.array([[[6, 5], [6, 6]], [[5, 5], [6, 6]]]).astype(np.float32)
    expected = model.predict(tensor_data)
    assert runner(onnx_model.graph.name, onnx_model, tensor_data, expected)


def test_two_zero_padding(runner):
    def my_func_1(input_layer, nf=64):
        temp = keras.layers.Conv2D(filters=nf, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_layer)
        pad1 = keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(temp)
        pad2 = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(temp)
        conv = keras.layers.Conv2D(filters=nf, kernel_size=(3, 3), strides=(1, 1), padding="valid")
        output = keras.layers.concatenate([conv(pad1), conv(pad2)], axis=3)
        return output

    input1 = Input(shape=(32, 32, 3))
    outputs = my_func_1(input1)
    model = keras.models.Model(inputs=input1, outputs=outputs)
    onnx_model = convert_keras(model, 'test_two_zero_padding')
    data = np.random.rand(1, 32, 32, 3).astype(np.float32)
    expected = model.predict(data)
    assert runner('test_two_zero_padding', onnx_model, data, expected)
