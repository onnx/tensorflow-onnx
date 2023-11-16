# SPDX-License-Identifier: Apache-2.0


"""Unit tests using onnx backends."""

import os
import unittest
from itertools import product

import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import Version
import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from backend_test_base import Tf2OnnxBackendTestBase
# pylint reports unused-wildcard-import which is false positive, __all__ is defined in common
from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tf2onnx import constants, utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.tf_loader import is_tf2, tf_placeholder_with_default, tf_placeholder
from tf2onnx.onnx_opset.signal import make_dft_constant

# pylint: disable=missing-docstring,invalid-name,unused-argument,function-redefined,cell-var-from-loop


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]

_STRIDE1x1 = [1, 1, 1, 1]
_KERNEL3x3 = [3, 3, 1, 1]
_DILATIONS1x1 = [1, 1, 1, 1]

# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFINPUT2 = "input2"
_INPUT2 = "input2:0"
_TFINPUT3 = "input3"
_INPUT3 = "input3:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"
_TFOUTPUT1 = "output1"
_OUTPUT1 = "output1:0"
_TFOUTPUT2 = "output2"
_OUTPUT2 = "output2:0"
_TFOUTPUT3 = "output3"
_OUTPUT3 = "output3:0"


if is_tf2():
    conv2d_backprop_input = tf.compat.v1.nn.conv2d_backprop_input
    conv3d_transpose = tf.compat.v1.nn.conv3d_transpose
    multinomial = tf.compat.v1.random.multinomial
    space_to_batch_nd = tf.compat.v1.space_to_batch_nd
    batch_to_space_nd = tf.compat.v1.batch_to_space_nd
    reverse_v2 = tf.compat.v1.reverse_v2
    random_normal = tf.compat.v1.random_normal
    random_uniform = tf.compat.v1.random_uniform
    fused_batch_norm = tf.compat.v1.nn.fused_batch_norm
    dropout = tf.compat.v1.nn.dropout
    resize_nearest_neighbor = tf.compat.v1.image.resize_nearest_neighbor
    quantize_and_dequantize = tf.quantization.quantize_and_dequantize
    resize_bilinear = tf.compat.v1.image.resize_bilinear
    resize_bilinear_v2 = tf.compat.v2.image.resize
    resize_area = tf.compat.v1.image.resize_area
    is_nan = tf.math.is_nan
    is_inf = tf.math.is_inf
    floormod = tf.math.floormod
    matrix_diag_part = tf.compat.v1.matrix_diag_part
    fake_quant_with_min_max_args = tf.quantization.fake_quant_with_min_max_args
    fake_quant_with_min_max_vars = tf.quantization.fake_quant_with_min_max_vars
elif Version(tf.__version__) >= Version("1.13"):
    conv2d_backprop_input = tf.compat.v1.nn.conv2d_backprop_input
    conv3d_transpose = tf.compat.v1.nn.conv3d_transpose
    multinomial = tf.compat.v1.random.multinomial
    space_to_batch_nd = tf.compat.v1.space_to_batch_nd
    batch_to_space_nd = tf.compat.v1.batch_to_space_nd
    reverse_v2 = tf.compat.v1.reverse_v2
    random_normal = tf.compat.v1.random_normal
    random_uniform = tf.compat.v1.random_uniform
    fused_batch_norm = tf.compat.v1.nn.fused_batch_norm
    dropout = tf.compat.v1.nn.dropout
    quantize_and_dequantize = tf.compat.v1.quantization.quantize_and_dequantize
    resize_area = tf.compat.v1.image.resize_area
    resize_nearest_neighbor = tf.compat.v1.image.resize_nearest_neighbor
    resize_bilinear = tf.compat.v1.image.resize_bilinear
    if Version(tf.__version__) >= Version("1.14"):
        resize_bilinear_v2 = tf.compat.v2.image.resize
    is_nan = tf.math.is_nan
    is_inf = tf.math.is_inf
    floormod = tf.floormod
    matrix_diag_part = tf.compat.v1.matrix_diag_part
    fake_quant_with_min_max_args = tf.compat.v1.quantization.fake_quant_with_min_max_args
    fake_quant_with_min_max_vars = tf.compat.v1.quantization.fake_quant_with_min_max_vars
else:
    conv2d_backprop_input = tf.nn.conv2d_backprop_input
    conv3d_transpose = tf.nn.conv3d_transpose
    multinomial = tf.multinomial
    space_to_batch_nd = tf.space_to_batch_nd
    batch_to_space_nd = tf.batch_to_space_nd
    reverse_v2 = tf.reverse_v2
    random_normal = tf.random_normal
    random_uniform = tf.random_uniform
    fused_batch_norm = tf.nn.fused_batch_norm
    dropout = tf.nn.dropout
    resize_nearest_neighbor = tf.image.resize_nearest_neighbor
    resize_bilinear = tf.image.resize_bilinear
    is_nan = tf.is_nan
    is_inf = tf.is_inf
    floormod = tf.floormod
    matrix_diag_part = tf.matrix_diag_part


def make_xval(shape):
    x_val = np.arange(np.prod(shape)).astype("float32").reshape(shape)
    return x_val


def get_conv_getdata(kind=1):
    if kind == 0:
        # generate all combinations (costly)
        dims = [
            ("padding", ["SAME", "VALID"]),
            ("input_sizes", [[32, 35, 35, 3], [32, 17, 17, 3], [1, 28, 28, 3], [32, 8, 8, 3]]),
            ("filter_sizes", [[1, 3, 3, 1], [1, 2, 2, 1], [1, 5, 5, 1], [1, 1, 1, 1], [1, 5, 2, 1], [1, 2, 5, 1]]),
            ("strides", [[1, 2, 2, 1], [1, 1, 1, 1]]),
        ]
        values = [key_values[1] for key_values in dims]
        for idx, v in enumerate(product(*values)):
            if True or idx == 30:
                yield (idx,) + v
    elif kind == 1:
        # some combination to that give decent padding coverage
        data = [
            ('SAME', [32, 35, 35, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 2, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 1, 1, 1], [1, 1, 1, 1]),
            ('SAME', [32, 35, 35, 3], [1, 5, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 2, 5, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 3], [1, 2, 5, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 2, 2, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 2, 2, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 5, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 5, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 8, 8, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [32, 8, 8, 3], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('VALID', [32, 35, 35, 3], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('VALID', [32, 35, 35, 3], [1, 2, 2, 1], [1, 2, 2, 1]),
        ]
        for idx, v in enumerate(data):
            yield (idx,) + v
    else:
        raise ValueError("kind not known")


def get_maxpoolwithargmax_getdata():
    data = [
        ('SAME', [1, 3, 3, 2], [1, 3, 3, 1], [1, 2, 2, 1]),
        ('SAME', [2, 5, 5, 3], [1, 4, 4, 1], [1, 2, 2, 1]),
        ('SAME', [2, 10, 5, 1], [1, 2, 2, 1], [1, 2, 2, 1]),
        ('SAME', [2, 10, 5, 3], [1, 4, 4, 1], [1, 1, 1, 1]),
        ('VALID', [2, 3, 3, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
        ('VALID', [2, 5, 5, 3], [1, 4, 4, 1], [1, 2, 2, 1]),
    ]
    for idx, v in enumerate(data):
        yield (idx,) + v


class BackendTests(Tf2OnnxBackendTestBase):
    def _run_test_case(self, func, output_names_with_port, feed_dict, **kwargs):
        kwargs["convert_var_to_const"] = False
        return self.run_test_case(func, feed_dict, [], output_names_with_port, **kwargs)

    def _test_expand_dims_known_rank(self, idx):
        x_val = make_xval([3, 4])
        def func(x):
            op = tf.expand_dims(x, idx)
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_expand_dims_known_rank(self):
        for i in [-1, 0, 1, -2]:
            self._test_expand_dims_known_rank(i)

    def test_expand_dims_one_unknown_rank(self):
        x_val = make_xval([3, 4])
        def func(x):
            op = tf.expand_dims(x, 0)
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_expand_dims_with_list(self):
        x_val = make_xval([3, 4])
        def func(x):
            op = tf.expand_dims(x, [[0]])
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def _test_expand_dims_more_unknown_rank(self, idx):
        x_val = make_xval([3, 4])
        def func(x):
            op = tf.expand_dims(x, idx)
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_expand_dims_more_unknown_rank(self):
        for i in [-1, 0, 1, -2]:
            self._test_expand_dims_more_unknown_rank(i)

    @check_opset_min_version(13, "Unsqueeze")
    def test_expand_dims_nonconst_dims(self):
        x_val = make_xval([3, 4])
        y_val = np.array([-1], dtype=np.int32)
        def func(x, y):
            op = tf.expand_dims(x, y)
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(9, "ConstantOfShape")
    def test_layer_normalization(self):
        x_val = make_xval([3, 4, 5])
        scale_val = make_xval([3, 4, 5]) * 0.2
        bias_val = make_xval([3, 4, 5]) * 0.1
        def func(x):
            mean = tf.reduce_mean(x, axis=[2], keepdims=True)
            centered = tf.subtract(x, mean)
            variance = tf.add(tf.reduce_mean(tf.square(centered), axis=[2], keepdims=True), 0.001)
            inv_std_dev = tf.math.rsqrt(variance)
            normalized = tf.multiply(centered, inv_std_dev)
            scaled = tf.multiply(normalized, scale_val)
            biased = tf.add(scaled, bias_val)
            return tf.identity(biased, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05,
                            graph_validator=lambda g: (check_op_count(g, "InstanceNormalization", 1)))

    @check_opset_min_version(9, "ConstantOfShape")
    def test_eye_non_const1(self):
        # tf.eye(num_rows), num_rows is not const here
        x_val = np.array(5, dtype=np.int32)
        def func(x):
            y = tf.eye(x, dtype=tf.int32)
            y1 = tf.eye(x, dtype=tf.int64)
            y2 = tf.eye(x, dtype=tf.float32)
            return tf.identity(y, name=_TFOUTPUT), tf.identity(y1, name=_TFOUTPUT1), tf.identity(y2, name=_TFOUTPUT2)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: x_val}, rtol=0)

        # tf.eye(num_rows, num_columns), both num_rows and num_columns are not const here
        x_val = np.array([5, 10], dtype=np.int32)
        def func(x):
            y = tf.eye(x[0], x[1], dtype=tf.int32)
            y1 = tf.eye(x[0], x[1], dtype=tf.int64)
            y2 = tf.eye(x[0], x[1], dtype=tf.float32)
            return tf.identity(y, name=_TFOUTPUT), tf.identity(y1, name=_TFOUTPUT1), tf.identity(y2, name=_TFOUTPUT2)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: x_val}, rtol=0)

    @check_tf_min_version("1.11", "eye has bug when version is below 1.11")
    @check_opset_min_version(9, "ConstantOfShape")
    def test_eye_non_const2(self):
        # tf.eye(num_rows), num_rows is not const here
        for np_dtype in [np.int32, np.int64, np.float32, np.float64]:
            x_val = np.array(5, dtype=np_dtype)
            def func(x):
                y = tf.eye(x, dtype=tf.int32)
                y1 = tf.eye(x, dtype=tf.float32)
                return tf.identity(y, name=_TFOUTPUT),\
                       tf.identity(y1, name=_TFOUTPUT1)
            self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val}, rtol=0)

        # tf.eye(num_rows, num_columns), both num_rows and num_columns are not const here
        for np_dtype in [np.int32, np.int64, np.float32, np.float64]:
            x_val = np.array([5, 10], dtype=np_dtype)
            def func(x):
                y = tf.eye(x[0], x[1], dtype=tf.int32)
                y1 = tf.eye(x[0], x[1], dtype=tf.float32)
                return tf.identity(y, name=_TFOUTPUT), \
                       tf.identity(y1, name=_TFOUTPUT1)
            self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val}, rtol=0)

    @check_opset_min_version(7, "trig")
    def test_trig_ops(self):
        for op in [tf.sin, tf.cos, tf.tan, tf.asin, tf.acos, tf.atan]:
            x_val = make_xval([3, 4])
            def func(x):
                op_ = op(x)
                return tf.identity(op_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-06)

    @check_opset_min_version(9, "trigh")
    def test_atrig_ops(self):
        for op in [tf.sinh, tf.cosh, tf.atanh, tf.asinh, tf.acosh]:
            x_val = make_xval([3, 4])
            def func(x):
                op_ = op(x)
                return tf.identity(op_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_opset_min_version(7, "multinomial")
    def test_multinomial(self):
        x_val = np.array([[10., 10.]], dtype=np.float32)
        def func(x):
            op = multinomial(tf.math.log(x), 5, output_dtype=tf.int32)
            return tf.identity(op, name=_TFOUTPUT)

        # since returned indexes are random we can only check type and shape
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_value=False,
                            check_shape=True, check_dtype=True)

    @skip_caffe2_backend()
    @check_opset_min_version(7, "multinomial")
    def test_multinomial1(self):
        shape = [2, 10]
        x_val = np.ones(np.prod(shape)).astype("float32").reshape(shape)
        def func(x):
            op = multinomial(x, 2, output_dtype=tf.int32)
            return tf.identity(op, name=_TFOUTPUT)
        # since returned indexes are random we can only check type and shape
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_value=False,
                            check_shape=True, check_dtype=True)

    def test_maxpool(self):
        for p in get_conv_getdata():
            _, padding, x_shape, ksize, strides = p
            x_val = make_xval(x_shape)
            def func(x):
                mp = tf.nn.max_pool(x, ksize, strides, padding=padding)
                return tf.identity(mp, name=_TFOUTPUT)
            self.logger.debug(str(p))
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.15", "required for max_pool args")
    def test_maxpool_int(self):
        x_shape = [8, 16, 16, 3]
        x_val = make_xval(x_shape).astype("int32")
        def func(x):
            mp = tf.nn.max_pool(x, ksize=[2], strides=[1, 2, 2, 1], padding="SAME")
            return tf.identity(mp, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tf_cpu("only tf_gpu can run maxpool with NCHW format")
    def test_maxpool_gpu(self):
        # make sure converter behaves well when data format is NCHW
        # and when data format is NCHW, only gpu version of tensorflow can run it.
        ksize = [1, 1, 2, 2]
        strides = [1, 1, 2, 2]
        x_val = make_xval([1, 3, 50, 80])
        for padding in ["SAME", "VALID"]:
            def func(x):
                mp = tf.nn.max_pool(x, ksize, strides, padding=padding, data_format="NCHW")
                return tf.identity(mp, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("AveragePool")
    def test_avgpool(self):
        for p in get_conv_getdata(kind=0):
            _, padding, x_shape, ksize, strides = p
            x_val = make_xval(x_shape)
            def func(x):
                mp = tf.nn.avg_pool(x, ksize, strides, padding=padding)
                return tf.identity(mp, name=_TFOUTPUT)

            self.logger.debug(str(p))
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-06)

    @check_onnxruntime_incompatibility("AveragePool")
    @skip_tf_cpu("only tf_gpu can run avgpool with NCHW format")
    def test_avgpool_gpu(self):
        ksize = [1, 1, 2, 2]
        strides = [1, 1, 2, 2]
        x_val = make_xval([1, 3, 50, 80])
        for padding in ["SAME", "VALID"]:
            def func(x):
                mp = tf.nn.avg_pool(x, ksize, strides, padding=padding, data_format="NCHW")
                return tf.identity(mp, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def _conv_test(self, x_val, w, strides=None, padding="VALID", dilations=None, rtol=1e-07):
        if strides is None:
            strides = _STRIDE1x1
        if dilations is None:
            dilations = _DILATIONS1x1
        def func(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv2d(x, kernel, strides=strides, padding=padding, dilations=dilations)
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=rtol)

    def test_conv2d_1(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape(_KERNEL3x3)
        self._conv_test(x_val, w)

    def test_conv2d_2(self):
        x_val = np.array([[4, 3, 1, 0],
                          [2, 1, 0, 1],
                          [1, 2, 4, 1],
                          [3, 1, 0, 2]], dtype=np.float32).reshape([1, 4, 4, 1])
        w = np.array([[1, 0, 1],
                      [2, 1, 0],
                      [0, 0, 1]], dtype=np.float32).reshape(_KERNEL3x3)
        self._conv_test(x_val, w)

    def test_conv2d_3(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape(_KERNEL3x3)
        self._conv_test(x_val, w)

    def test_conv2d_4(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.random.random_sample(_KERNEL3x3).astype(np.float32)
        self._conv_test(x_val, w, padding="SAME", rtol=1e-05)

    def test_conv2d_5(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        kernel_shape = [3, 3, 1, 2]
        w = np.random.random_sample(kernel_shape).astype(np.float32)
        self._conv_test(x_val, w, padding="SAME", rtol=1e-05)

    def test_conv2d_6(self):
        x_shape = [1, 35, 35, 288]  # out: [1, 17, 17, 384]
        kernel_shape = [3, 3, 288, 384]
        strides = [1, 2, 2, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        self._conv_test(x_val, kernel_val, strides=strides, padding="VALID", rtol=1.1e-05)

    @check_tf_min_version("1.14", "tf 1.14 needed for explicit padding")
    def test_conv2d_explicit_padding(self):
        x_shape = [1, 35, 35, 288]
        kernel_shape = [3, 3, 288, 384]
        pads = [[0, 0], [1, 2], [3, 4], [0, 0]]
        strides = [1, 1, 1, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        self._conv_test(x_val, kernel_val, strides=strides, padding=pads, rtol=1.1e-05)

    def test_conv2d_dilation_same(self):
        x_shape = [1, 35, 35, 288]  # NHWC
        kernel_shape = [3, 3, 288, 384]  # [filter_height, filter_width, in_channels, out_channels]
        strides = [1, 1, 1, 1]  # NHWC
        dilations = [1, 3, 1, 1]  # NHWC
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        self._conv_test(x_val, kernel_val, strides=strides, padding="SAME", dilations=dilations, rtol=1.1e-05)

    def test_conv2d_dilation_strides_same(self):
        x_shape = [1, 35, 35, 288]  # NHWC
        kernel_shape = [3, 3, 288, 384]  # [filter_height, filter_width, in_channels, out_channels]
        strides = [1, 2, 4, 1]  # NHWC
        dilations = [1, 3, 1, 1]  # NHWC
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        self._conv_test(x_val, kernel_val, strides=strides, padding="SAME", dilations=dilations, rtol=1e-05)

    def test_conv3d_1(self):
        strides = [1, 1, 1, 1, 1]
        dilations = [1, 1, 1, 1, 1]
        x_val = np.random.random_sample([2, 10, 9, 8, 5]).astype(np.float32)
        w = np.random.random_sample([2, 3, 4, 5, 6]).astype(np.float32)
        padding = "VALID"
        def func(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv3d(x, kernel, strides=strides, padding=padding, data_format="NDHWC", dilations=dilations)
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    def test_conv3d_2(self):
        strides = [1, 2, 3, 1, 1]
        dilations = [1, 1, 1, 1, 1]
        x_val = np.random.random_sample([2, 10, 9, 8, 5]).astype(np.float32)
        w = np.random.random_sample([2, 3, 4, 5, 6]).astype(np.float32)
        padding = "VALID"
        def func(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv3d(x, kernel, strides=strides, padding=padding, data_format="NDHWC", dilations=dilations)
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    def test_conv3d_3(self):
        strides = [1, 2, 3, 1, 1]
        dilations = [1, 1, 1, 1, 1]
        x_val = np.random.random_sample([2, 10, 9, 8, 5]).astype(np.float32)
        w = np.random.random_sample([2, 3, 4, 5, 6]).astype(np.float32)
        padding = "SAME"
        def func(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv3d(x, kernel, strides=strides, padding=padding, data_format="NDHWC", dilations=dilations)
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    def test_avgpool3d(self):
        strides = [1, 1, 1, 1, 1]
        ksize = [1, 2, 2, 3, 1]
        x_val = np.random.random_sample([2, 10, 9, 8, 5]).astype(np.float32)
        padding = "VALID"

        def func(x):
            mp = tf.nn.avg_pool3d(x, ksize, strides, padding=padding, data_format="NDHWC")
            return tf.identity(mp, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_maxpool3d(self):
        strides = [1, 1, 1, 1, 1]
        ksize = [1, 2, 2, 3, 1]
        x_val = np.random.random_sample([2, 10, 9, 8, 5]).astype(np.float32)
        padding = "VALID"

        def func(x):
            mp = tf.nn.max_pool3d(x, ksize, strides, padding=padding, data_format="NDHWC")
            return tf.identity(mp, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14", "tf.nn.avg_pool2d doesn't exist before tf 1.14")
    def test_avgpool2d(self):
        strides = [1, 1, 1, 1]
        ksize = [1, 2, 3, 1]
        x_val = make_xval([2, 10, 12, 3])
        padding = "VALID"

        def func(x):
            mp = tf.nn.avg_pool2d(x, ksize, strides, padding=padding, data_format="NHWC")
            return tf.identity(mp, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})


    @check_tf_min_version("1.7", "tf only support dilation is 1 for now")
    def test_conv2d_7(self):
        x_shape = [1, 35, 35, 288]  # out: [1, 17, 17, 384]
        kernel_shape = [3, 3, 288, 384]
        strides = [1, 2, 2, 1]
        dilations = [1, 3, 3, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        self._conv_test(x_val, kernel_val, strides=strides, padding="VALID",
                        dilations=dilations, rtol=1e-05)

    def test_conv2d_8(self):
        for input_shape in [[10, 10], [5, 5]]:
            x_val = make_xval((1, 1, *input_shape)).transpose(NCHW_TO_NHWC)
            w = np.random.random_sample([3, 3, 1, 2]).astype(np.float32)
            strides = [1, 2, 2, 1]
            def func(x):
                kernel = tf.constant(w, dtype=tf.float32, name='k')
                conv = tf.nn.conv2d(x, kernel, strides=strides, padding="SAME")
                return tf.identity(conv, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-5)

    def test_conv2d_with_pad_valid(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.random.random_sample([3, 3, 1, 2]).astype(np.float32)
        strides = [1, 1, 1, 1]
        def func(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            x_pad = tf.pad(x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])
            conv = tf.nn.conv2d(x_pad, kernel, strides=strides, padding="VALID")
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-5)

    def test_conv2d_with_pad_same(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.random.random_sample([3, 3, 1, 2]).astype(np.float32)
        strides = [1, 1, 1, 1]
        def func(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            x_pad = tf.pad(x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])
            conv = tf.nn.conv2d(x_pad, kernel, strides=strides, padding="SAME")
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-5)

    def test_conv2d_transpose(self):
        x_shape = [2, 6, 4, 3]
        output_shape = [2, 13, 9, 2]
        kernel_shape = [3, 3, 2, 3]
        strides = [1, 2, 2, 1]
        x_val = make_xval(x_shape)
        kernel_val = make_xval(kernel_shape)
        def func(x):
            f = tf.constant(kernel_val, name="kernel", dtype=tf.float32)
            conv = tf.nn.conv2d_transpose(x, f, output_shape, strides=strides, padding="VALID")
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_min_version("0.5.0", "conv transpose is added since onnxruntime-0.5.0")
    def test_conv2d_transpose2(self):
        # output_shape is dynamic
        extra_opset = [utils.make_opsetid(constants.MICROSOFT_DOMAIN, 1)]
        process_args = {"extra_opset": extra_opset}
        x_shape = [2, 6, 4, 3]
        output_shape = np.array([2, 13, 9, 2]).astype(np.int32)
        kernel_shape = [3, 3, 2, 3]
        strides = [1, 2, 2, 1]
        x_val = make_xval(x_shape)
        kernel_val = make_xval(kernel_shape)
        def func(x, output_shape_placeholder):
            f = tf.constant(kernel_val, name="kernel", dtype=tf.float32)
            conv = tf.nn.conv2d_transpose(x, f, output_shape_placeholder, strides=strides, padding="VALID")
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: output_shape},
                            rtol=1e-05, process_args=process_args)

    @check_opset_min_version(10, "quantize_and_dequantize")
    def test_conv2d_quantization_axis(self):
        x_shape = [1, 1, 5, 5]
        kernel_shape = _KERNEL3x3
        strides = [1, 1, 1, 1]
        x_val = make_xval(x_shape).transpose(NCHW_TO_NHWC)
        kernel_val = make_xval(_KERNEL3x3)

        def func(x):
            f = tf.constant(kernel_val, name="kernel", dtype=tf.float32)
            kernel_dq = quantize_and_dequantize(f, 0, np.prod(kernel_shape))
            conv = tf.nn.conv2d(x, kernel_dq, strides=strides, padding="VALID")
            return tf.identity(conv, name=_TFOUTPUT)
        def graph_validator(g):
            return check_quantization_axis(g, "DequantizeLinear", 0)

        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05, graph_validator=graph_validator,
                            check_shape=False)

    @check_opset_min_version(10, "quantize_and_dequantize")
    def test_conv2d_transpose_quantization_axis(self):
        x_shape = [2, 6, 4, 3]
        output_shape = [2, 13, 9, 2]
        kernel_shape = [3, 3, 2, 3]
        strides = [1, 2, 2, 1]
        x_val = make_xval(x_shape)
        kernel_val = make_xval(kernel_shape)
        def func(x):
            f = tf.constant(kernel_val, name="kernel", dtype=tf.float32)
            kernel_dq = quantize_and_dequantize(f, 0, np.prod(kernel_shape))
            conv = tf.nn.conv2d_transpose(x, kernel_dq, output_shape, strides=strides, padding="VALID")
            return tf.identity(conv, name=_TFOUTPUT)
        def graph_validator(g):
            return check_quantization_axis(g, "DequantizeLinear", 1)

        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05, graph_validator=graph_validator,
                            check_shape=False)

    def test_depthwiseconv_0(self):
        x_shape = [1, 3, 4, 3]
        kernel_shape = [3, 3, 3, 3]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        def func(x):
            kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
            conv = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=0.08)

    def test_depthwiseconv_1(self):
        x_shape = [1, 112, 112, 32]
        kernel_shape = [3, 3, 32, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        def func(x):
            kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
            conv = tf.nn.depthwise_conv2d(x, kernel, strides=_STRIDE1x1, padding='VALID')
            return tf.identity(conv, name=_TFOUTPUT)
        # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=0.08)

    def test_depthwiseconv_3(self):
        x_shape = [1, 112, 112, 32]
        kernel_shape = [3, 3, 32, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        def func(x):
            kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
            conv = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
            return tf.identity(conv, name=_TFOUTPUT)
        # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=0.01)

    def test_depthwiseconv_shared_kernel(self):
        x_shape = [1, 3, 4, 3]
        kernel_shape = [3, 3, 3, 3]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        def func(x, y):
            kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
            conv1 = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
            conv2 = tf.nn.depthwise_conv2d(y, kernel, strides=[1, 1, 1, 1], padding='VALID')
            conv = tf.add(conv1, conv2)
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: x_val}, rtol=0.08)

    @check_tf_min_version("1.14", "tf depthwise_conv2d dilations")
    @check_opset_min_version(11, "non-const pads")
    def test_depthwiseconv_dilations(self):
        x_shape = [1, 32, 32, 1]
        kernel_shape = [5, 5, 1, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        def func(x):
            kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
            conv = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME', dilations=[3, 4])
            return tf.identity(conv, name=_TFOUTPUT)
        # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=0.01)

    @check_tf_max_version("1.15", "not supported in tf-2.0")
    def test_dropout(self):
        x_val = np.ones([1, 24, 24, 3], dtype=np.float32)
        # Define a scope for reusing the variables
        def func(x):
            is_training = tf.constant(False, tf.bool)
            x_ = tf.identity(x)
            fc1 = tf.layers.dropout(x_, rate=.1, training=is_training)
            return tf.identity(fc1, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val},
                            graph_validator=lambda g: (check_op_count(g, "RandomUniform", 0) and
                                                       check_op_count(g, "RandomUniformLike", 0)))

    def test_nn_dropout(self):
        x_val = np.ones([1, 24, 24, 3], dtype=np.float32)
        # Define a scope for reusing the variables
        def func(x, keep_prob):
            x_ = tf.identity(x)
            fc1 = dropout(x_, keep_prob)
            return tf.identity(fc1, name=_TFOUTPUT)
        # when constant_fold is enabled, PlaceholderWithDefault will be folded into either a const or a placeholder.
        # here we set it False to test PlaceholderWithDefault bug: https://github.com/onnx/tensorflow-onnx/pull/446
        # Dropout with ratio 1.0 will be optimized so that only one Identity is left
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: np.array(1., dtype=np.float32)},
                            graph_validator=lambda g: (check_op_count(g, "RandomUniform", 0) and
                                                       check_op_count(g, "RandomUniformLike", 0)))

    @check_tf_min_version("1.13")
    def test_nn_dropout_with_rate(self):
        rate = tf.constant(0., name="rate")
        x_val = np.ones([1, 24, 24, 3], dtype=np.float32)
        # Define a scope for reusing the variables
        def func(x):
            x_ = tf.identity(x)
            fc1 = tf.nn.dropout(x_, rate=rate)
            return tf.identity(fc1, name="output")
        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port,
                           graph_validator=lambda g: (check_op_count(g, "RandomUniform", 0) and
                                                      check_op_count(g, "RandomUniformLike", 0)))

    def test_inputs_as_nchw_arg(self):
        x_shape = [2, 32, 32, 3]
        kernel_shape = [3, 3, 3, 3]
        x_val = make_xval(x_shape)
        x_val_for_onnx = x_val.transpose(NHWC_TO_NCHW)
        def func(x):
            kernel = tf.constant(make_xval(kernel_shape), dtype=tf.float32, name='k')
            conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05,
                            process_args={"inputs_as_nchw": [_INPUT]},
                            onnx_feed_dict={_INPUT: x_val_for_onnx})

    def test_outputs_as_nchw_arg(self):
        x_shape = [2, 32, 32, 3]
        kernel_shape = [3, 3, 3, 3]
        x_val = make_xval(x_shape)
        def func(x):
            kernel = tf.constant(make_xval(kernel_shape), dtype=tf.float32, name='kernel')
            conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05,
                            process_args={"outputs_as_nchw": [_OUTPUT]})

    @skip_tflite("TFlite adds ops that obscure pattern")
    @check_tf_min_version("1.15")
    def test_conv1d_dilations_rewriter(self):
        x_shape = [2, 32, 3]
        x_val = make_xval(x_shape)
        for p in ['SAME', 'VALID']:
            def func(x):
                t = tf.keras.layers.Conv1D(filters=768, kernel_size=3, dilation_rate=3, padding=p)
                t.build(x_shape)
                y = t.call(x)
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @check_tf_min_version("1.15")
    @skip_tf_cpu("only tf_gpu can run conv2d with NCHW format")
    def test_conv2d_biasadd_rewriter(self):
        x_shape = [2, 3, 32, 16]
        x_val = make_xval(x_shape)
        def func(x):
            middles = tf.keras.layers.ZeroPadding2D(
                padding=(0, 4),
                data_format="channels_first",
                name="padding"
            )(x)
            t = tf.keras.layers.Conv2D(
                filters=768,
                kernel_size=3,
                strides=1,
                use_bias=True,
                data_format="channels_first",
                name="conv2d"
            )(middles)
            return tf.identity(t, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                            graph_validator=lambda g: check_op_count(g, "Add", 0, disabled=False))

    @check_tf_min_version("1.15")
    def test_conv2d_dilations_rewriter(self):
        x_shape = [2, 32, 16, 3]
        x_val = make_xval(x_shape)
        for p in ['SAME', 'VALID']:
            def func(x):
                t = tf.keras.layers.Conv2D(filters=768, kernel_size=3, dilation_rate=3, padding=p)
                t.build(x_shape)
                y = t.call(x)
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))
            def func(x):
                t = tf.keras.layers.DepthwiseConv2D(kernel_size=3, dilation_rate=3, padding=p)
                t.build(x_shape)
                y = t.call(x)
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @check_tf_min_version("1.15")
    @skip_tf_cpu("only tf_gpu can run conv2d with NCHW format")
    def test_nchw_conv2d_dilations_rewriter(self):
        x_shape = [2, 3, 32, 16]
        x_val = make_xval(x_shape)
        for p in ['SAME', 'VALID']:
            def func(x):
                t = tf.keras.layers.Conv2D(
                    filters=768,
                    kernel_size=3,
                    dilation_rate=3,
                    padding=p,
                    data_format='channels_first'
                )
                t.build(x_shape)
                y = t.call(x)
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))
            def func(x):
                t = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3,
                    dilation_rate=3,
                    padding=p,
                    data_format='channels_first'
                )
                t.build(x_shape)
                y = t.call(x)
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @check_tf_min_version("1.15")
    @skip_tflite("TFlite adds ops that obscure pattern")
    @allow_missing_shapes("Rewriting makes some shapes known")
    def test_conv2d_dilations_rewriter_unknown_shape(self):
        x_shape = [2, 32, 16, 3]
        x_val = make_xval(x_shape)
        def func():
            x = tf_placeholder(tf.float32, [2, None, None, 3], name=_TFINPUT)
            t = tf.keras.layers.Conv2D(filters=768, kernel_size=3, dilation_rate=3, padding="VALID")
            t.build(x_shape)
            y = t.call(x)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2,
                            as_session=True, premade_placeholders=True,
                            graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @check_tf_min_version("1.15")
    @skip_tflite("TFlite adds ops that obscure pattern")
    @skip_tf_cpu("only tf_gpu can run conv2d with NCHW format")
    @allow_missing_shapes("Rewriting makes some shapes known")
    def test_nchw_conv2d_dilations_rewriter_unknown_shape(self):
        x_shape = [2, 3, 32, 16]
        x_val = make_xval(x_shape)
        def func():
            x = tf_placeholder(tf.float32, [2, 3, None, None], name=_TFINPUT)
            t = tf.keras.layers.Conv2D(
                filters=768,
                kernel_size=3,
                dilation_rate=3,
                padding="VALID",
                data_format='channels_first'
            )
            t.build(x_shape)
            y = t.call(x)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2,
                            as_session=True, premade_placeholders=True,
                            graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @check_tf_min_version("1.15")
    def test_conv3d_dilations_rewriter(self):
        x_shape = [2, 32, 16, 8, 3]
        x_val = make_xval(x_shape)
        for p in ['SAME', 'VALID']:
            def func(x):
                t = tf.keras.layers.Conv3D(filters=768, kernel_size=3, dilation_rate=3, padding=p)
                t.build(x_shape)
                y = t.call(x)
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @check_tf_min_version("1.15")
    @skip_tf_cpu("only tf_gpu can run conv3d with NCDHW format")
    def test_ncdhw_conv3d_dilations_rewriter(self):
        x_shape = [2, 3, 32, 16, 8]
        x_val = make_xval(x_shape)
        for p in ['SAME', 'VALID']:
            def func(x):
                t = tf.keras.layers.Conv3D(
                    filters=768,
                    kernel_size=3,
                    dilation_rate=3,
                    padding=p,
                    data_format='channels_first'
                )
                t.build(x_shape)
                y = t.call(x)
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @skip_tf2("Uses tf.layers")
    def test_conv1d_tf1_dilations_rewriter(self):
        x_shape = [2, 32, 3]
        x_val = make_xval(x_shape)
        for p in ['SAME', 'VALID']:
            def func(x):
                y = tf.layers.conv1d(x, filters=768, kernel_size=3, dilation_rate=3, padding=p, name="conv1")
                return tf.identity(y, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2, as_session=True,
                                graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    @skip_tf2("Uses tf.layers")
    def test_conv1d_tf1_dilations_rewriter_unknown_shape(self):
        x_shape = [2, 32, 3]
        x_val = make_xval(x_shape)
        def func():
            x = tf_placeholder(tf.float32, [2, None, 3], name=_TFINPUT)
            y = tf.layers.conv1d(x, filters=768, kernel_size=3, dilation_rate=3, padding="VALID", name="conv1")
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04, atol=1e-2,
                            as_session=True, premade_placeholders=True,
                            graph_validator=lambda g: check_op_count(g, "Reshape", 0, disabled=False))

    def test_lrn_default(self):
        x_shape = [1, 3, 4, 3]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x):
            op = tf.nn.local_response_normalization(x)
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    def test_lrn(self):
        # can't set bias = 0
        x_shape = [1, 2, 2, 8]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x):
            op = tf.nn.local_response_normalization(x, depth_radius=4, bias=2, alpha=2, beta=1)
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_incompatibility("Abs")
    def test_abs(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.abs(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Add")
    def test_const(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            y = tf.constant(x_val, name="y")
            return tf.add(x, y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Add")
    def test_add(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.add(x, x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_placeholder(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            return tf.identity(x, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_placeholder_with_default_use_default(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func():
            x = tf.constant(x_val, name="x")
            y = tf_placeholder_with_default(x, x_val.shape, name=_TFINPUT)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, as_session=True, premade_placeholders=True)

    def test_placeholder_with_default_use_feed(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func():
            x = tf.constant(x_val, name="x")
            y = tf_placeholder_with_default(x, x_val.shape, name=_TFINPUT)
            return tf.identity(y, name=_TFOUTPUT)
        x_feed_val = np.array([11.0, 22.0, -33.0, -44.0], dtype=np.float32).reshape((2, 2))
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_feed_val}, as_session=True, premade_placeholders=True)

    def test_placeholder_with_default_computed_use_default(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        y_val = np.array([2.0, -4.0, 6.0, -8.0], dtype=np.float32).reshape((2, 2))
        def func():
            x = tf_placeholder(tf.float32, x_val.shape, name=_TFINPUT)
            y = tf_placeholder(tf.float32, y_val.shape, name=_TFINPUT1)
            total = tf.add(x, y)
            z = tf_placeholder_with_default(total, x_val.shape, name=_TFINPUT2)
            total2 = tf.add(total, z)
            return tf.identity(total2, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val}, as_session=True,
                            premade_placeholders=True, process_args={'use_default': [_TFINPUT2]})

    def test_placeholder_with_default_computed_ignore_default(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        y_val = np.array([2.0, -4.0, 6.0, -8.0], dtype=np.float32).reshape((2, 2))
        z_val = np.array([3.0, 6.0, 9.0, 10.0], dtype=np.float32).reshape((2, 2))
        def func():
            x = tf_placeholder(tf.float32, x_val.shape, name=_TFINPUT)
            y = tf_placeholder(tf.float32, y_val.shape, name=_TFINPUT1)
            total = tf.add(x, y)
            z = tf_placeholder_with_default(total, x_val.shape, name=_TFINPUT2)
            total2 = tf.add(total, z)
            return tf.identity(total2, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val}, as_session=True,
                            premade_placeholders=True, process_args={'ignore_default': [_TFINPUT2]})

    def test_fold_cond_keras_learning_phase(self):
        # keras_learning_phase can slip into frozen graphs and cause huge inefficiencies with If nodes.
        # Should be removed and Ifs folded.
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func():
            x = tf_placeholder(tf.float32, [None, None], name=_TFINPUT)
            learning_phase = tf_placeholder_with_default(False, [], name="keras_learning_phase")
            y = tf.cond(learning_phase, lambda: x * 2, lambda: x * 3)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, as_session=True, premade_placeholders=True,
                            graph_validator=lambda g: check_op_count(g, "If", 0, disabled=False))

    @check_onnxruntime_incompatibility("Add")
    def test_add_bcast(self):
        x1_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x2_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32).reshape((2, 2, 2))
        def func(x1, x2):
            x_ = tf.add(x1, x2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x1_val, _INPUT1: x2_val})

    @check_onnxruntime_incompatibility("Add")
    def test_add_bcast1(self):
        # example taken from onnx doc
        x1_val = np.random.randn(3, 4, 5).astype(np.float32)
        x2_val = np.random.randn(5).astype(np.float32)
        def func(x1, x2):
            x_ = tf.add(x1, x2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x1_val, _INPUT1: x2_val})

    def test_matmul0(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.matmul(x, x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("Issue with matmul with 2 copies of same input")
    def test_matmul1(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0, 5.0, 6.0], dtype=np.float32).reshape((2, 3))
        def func(x):
            x_ = tf.matmul(x, x, transpose_a=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_matmul2(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        y_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x, y):
            x_ = tf.matmul(x, y, transpose_b=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @unittest.skipIf(get_test_config().is_mac and get_test_config().is_onnxruntime_backend
                     and get_test_config().backend_version == "0.2.1", "onnxruntime 0.2.1 has bug on mac")
    def test_matmul3(self):
        x_shape = [1, 12, 256, 64]
        x_val = np.arange(np.prod(x_shape)).astype("float32").reshape((x_shape))
        def func(x, y):
            x_ = tf.matmul(x, y, transpose_b=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: x_val}, rtol=1e-5)

    @check_tf_min_version("2.6")
    def test_matmulinteger(self):
        x_val = np.array([1, 2, -3, -4], dtype=np.int8).reshape((2, 2))
        y_val = np.array([1, 2, -3, -4], dtype=np.int8).reshape((2, 2))
        def func(x, y):
            x_ = tf.matmul(x, y, output_type=tf.int32)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_onnxruntime_incompatibility("Sub")
    def test_sub(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.subtract(x, x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Mul")
    def test_multiply(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.multiply(x, x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Div")
    def test_div(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.realdiv(x, x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "float equality")
    def test_div_no_nan(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0, 5.0, 0.0, float("nan"), float("-inf"), float("inf")], dtype=np.float32)
        y_val = np.array([1.0, 0.5, 0.0, -4.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float32)
        def func(x, y):
            x_ = tf.math.divide_no_nan(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        # TFLite expresses infinity as a value > 1e38
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val}, mtol=1e38)

    @check_onnxruntime_incompatibility("Exp")
    def test_exp(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.exp(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_incompatibility("Log")
    def test_log(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.math.log(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Log")
    def test_log_double(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64).reshape((2, 2))
        def func(x):
            x_ = tf.math.log(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_gather(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        idx = np.array([1, 0, 2], dtype=np.int32)
        idx_flattened = np.array([i * x_val.shape[1] + idx for i in range(0, x_val.shape[0])])
        def func(x):
            x_ = tf.gather(tf.reshape(x, [-1]), tf.constant(idx_flattened))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(12, "GatherND with batch_dims")
    def test_gather_batch_dims_no_trans(self):
        x_val = np.arange(2 * 2 * 3 * 5 * 4, dtype=np.float32).reshape((2, 2, 3, 5, 4))
        idx_val = np.array([[[1, 0, 2, 0], [1, 1, 1, 0]], [[0, 0, 0, 0], [2, 1, 1, 0]]], dtype=np.int32)
        def func(x, idx):
            x_ = tf.gather(x, idx, batch_dims=2, axis=2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: idx_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(12, "GatherND with batch_dims")
    def test_gather_batch_dims(self):
        x_val = np.arange(2 * 2 * 3 * 5 * 4, dtype=np.float32).reshape((2, 2, 3, 5, 4))
        idx_val = np.array([[[1, 0, 2, 0], [1, 1, 1, 0]], [[0, 0, 0, 0], [2, 1, 1, 0]]], dtype=np.int32)
        def func(x, idx):
            x_ = tf.gather(x, idx, batch_dims=2, axis=3)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: idx_val})

    @check_opset_min_version(10, "Slice")
    def test_roll_axis_scalar(self):
        x_val = np.arange(4 * 3 * 5 * 2, dtype=np.float32).reshape((4, 3, 5, 2))
        shift_val = np.array(4, dtype=np.int64)
        axes_val = np.array(2, dtype=np.int32)
        def func(x, shift):
            x_ = tf.roll(x, shift, axes_val)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: shift_val})

    @check_opset_min_version(10, "Slice")
    def test_roll_axis_vector(self):
        x_val = np.arange(4 * 3 * 5 * 2, dtype=np.float32).reshape((4, 3, 5, 2))
        shift_val = np.array([2, 3, 4], dtype=np.int32)
        axes_val = np.array([1, 2, 1], dtype=np.int32)
        def func(x, shift):
            x_ = tf.roll(x, shift, axes_val)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: shift_val})

    @check_opset_min_version(10, "Slice")
    def test_roll_neg_axis(self):
        def func(input_ids):
            shifted_input_ids = tf.cast(input_ids, tf.int32)
            shifted_input_ids = tf.roll(shifted_input_ids, 1, axis=-1)
            return tf.identity(shifted_input_ids, name=_TFOUTPUT)
        x_val = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_roll_neg_shift(self):
        x_val = np.arange(4 * 3 * 5 * 2, dtype=np.float32).reshape((4, 3, 5, 2))
        shift_val = np.array([-2, 13, -3], dtype=np.int32)
        axes_val = np.array([1, 2, -1], dtype=np.int32)
        def func(x, shift):
            x_ = tf.roll(x, shift, axes_val)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: shift_val})

    @check_tf_min_version("2.2")
    def test_large_model_format(self):
        x_val = np.array([2.0], dtype=np.float32)
        y_const = np.arange(2000, dtype=np.float32)
        def func(x):
            x_ = tf.multiply(x, tf.constant(y_const))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, large_model=True)

    @check_target('rs6', 'GatherNd')
    def test_gathernd(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        indices = np.array([[[0, 1], [1, 1]], [[1, 2], [0, 2]]], dtype=np.int32)
        def func(x):
            x_ = tf.gather_nd(x, tf.constant(indices))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        indices = np.array([[[0], [2]], [[4], [7]], [[6], [1]]], dtype=np.int32)
        def func(x):
            x_ = tf.gather_nd(x, tf.constant(indices))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_target('rs6', 'GatherNd')
    def test_gathernd_less_index(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        indices = np.array([[[0], [1]], [[2], [0]]], dtype=np.int32)
        def func(x):
            x_ = tf.gather_nd(x, tf.constant(indices))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        # shape: 2*2*2
        x_val = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        indices = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], dtype=np.int32)
        def func(x):
            x_ = tf.gather_nd(x, tf.constant(indices))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_opset_min_version(7, "tile")
    def test_tile(self):
        x_val = np.array([[0, 1], [2, 3]], dtype=np.float32)
        def func(x):
            multiple = tf.constant([2, 2])
            x_ = tf.tile(x, multiple)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "tile")
    def test_tile_const(self):
        # Should be folded
        x_val = np.array([[0, 1], [2, 3]], dtype=np.float32)
        def func():
            multiple = tf.constant([1000, 2])
            x_ = tf.tile(tf.constant(x_val), multiple)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, graph_validator=lambda g: check_op_count(g, "Tile", 0, disabled=False))

    @check_opset_min_version(7, "tile")
    def test_tile_large_const(self):
        # Should not be folded since it is so large
        x_val = np.array([[0, 1], [2, 3]], dtype=np.float32)
        def func():
            multiple = tf.constant([1000000, 2])
            x_ = tf.tile(tf.constant(x_val), multiple)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, graph_validator=lambda g: check_op_count(g, "Tile", 1, disabled=False))

    @check_onnxruntime_incompatibility("Neg")
    def test_neg(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.negative(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Mul")
    def test_square(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.square(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Min")
    def test_min(self):
        x_val1 = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.minimum(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

        x_val1 = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.int32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.minimum(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @skip_caffe2_backend("issue with broadcasting scalar")
    @check_onnxruntime_incompatibility("Sub")
    def test_min_broadcast(self):
        # tests if the broadcast for min/max is working
        x_val1 = np.array([2.0, 16.0, 5.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0], dtype=np.float32)
        def func(x1):
            x2 = tf.constant(x_val2, dtype=tf.float32, name='x2')
            mi = tf.minimum(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1})

    @check_onnxruntime_incompatibility("Add")
    def test_logicaland(self):
        x_val1 = np.array([1, 0, 1, 1], dtype=bool).reshape((2, 2))
        x_val2 = np.array([0, 1, 1, 1], dtype=bool).reshape((2, 2))
        def func(x1, x2):
            mi = tf.logical_and(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Greater")
    def test_greater(self):
        for op in [tf.greater, tf.greater_equal]:
            x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
            x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
            def func(x1, x2):
                mi = op(x1, x2)
                return tf.identity(mi, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Greater")
    def test_greater_unsupport_type(self):
        for op in [tf.greater, tf.greater_equal]:
            x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
            x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
            def func(x1, x2):
                mi = op(x1, x2)
                return tf.identity(mi, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Less")
    def test_less(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.less(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Less")
    def test_less_unsupport_type(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.less(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_opset_min_version(11, "Equal")
    def test_equal_float(self):
        x_val1 = np.array([0., 1., 2., 3., 4., -1., -2], dtype=np.float32)
        x_val2 = np.array([0., 1., 2.1, 3.5, 4.6, -1.1, -2.9], dtype=np.float32)
        def func(x1, x2):
            mi = tf.equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    def test_equal_string(self):
        x_val1 = np.array(['1'], dtype=np.string_)
        x_val2 = np.array(['2'], dtype=np.string_)
        def func(x1, x2):
            mi = tf.equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    def test_equal(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

        x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    def test_not_equal(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.not_equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

        x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
        def func(x1, x2):
            mi = tf.not_equal(x1, x2)
            return tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    def test_sequeeze_no_axis_specified(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 1, 2, 1, 1))
        def func(x):
            x_ = tf.squeeze(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_sequeeze_no_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.squeeze(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Pad")
    def test_sequeeze_no_axis_specified_unknown_rank(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_val = np.array([2, 1, 2, 1, 1], dtype=np.int64)
        z_val = np.zeros((1, 2), dtype=np.int64)
        def func(x, y, z):
            y_ = tf.pad(y, z)
            x_ = tf.reshape(x, y_)
            x_ = tf.squeeze(x_)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    def test_sequeeze_positive_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2, 1))
        def func(x):
            x_ = tf.squeeze(x, [2])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_sequeeze_negative_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2, 1))
        def func(x):
            x_ = tf.squeeze(x, [-1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_sequeeze_mixed_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((1, 2, 2, 1))
        def func(x):
            x_ = tf.squeeze(x, [0, -1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Squeeze")
    def test_sequeeze_mixed_axis_unknown_rank(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_val = np.array([2, 1, 2, 1, 1], dtype=np.int64)
        z_val = np.zeros((1, 2), dtype=np.int64)
        def func(x, y, z):
            y_ = tf.pad(y, z)
            x_ = tf.reshape(x, y_)
            x_ = tf.squeeze(x_, [1, -1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    def test_transpose(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape((2, 3))
        def func(x):
            x_ = tf.transpose(x)  # perm=[1,0])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_reshape(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            shape = tf.constant([1, 4])
            x_ = tf.reshape(x, shape)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_shape=True)

    def test_reshape_reshape(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            shape = tf.constant([1, 4])
            shape_2 = tf.constant([4, 1])
            x_ = tf.reshape(x, shape)
            x_ = tf.reshape(x_, shape_2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val},
                            graph_validator=lambda g: check_op_count(g, "Reshape", 1, disabled=False))

    @check_opset_min_version(6, "cast")
    def test_reshape_int(self):
        x_val = np.array([1, 2, 3, 4], dtype=np.int32).reshape((2, 2))
        def func(x):
            shape = tf.constant([1, 4])
            x_ = tf.reshape(x, shape)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_shape=True)

    @check_opset_min_version(6, "cast")
    def test_reshape_dynamic(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        shape_val = np.array([4, 1], dtype=np.int32)
        def func(x, shape):
            x_ = tf.reshape(x, shape)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: shape_val}, check_shape=True)

    @check_onnxruntime_incompatibility("Relu")
    def test_relu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.nn.relu(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("fails on caffe2 with dim issue")
    @check_onnxruntime_incompatibility("Mul")
    @check_tf_min_version("1.6")
    def test_leaky_relu_int(self):
        # starting from tf 1.6, leaky_relu supports `feature` x of int type
        x_types = [np.int32, np.int64]
        for x_type in x_types:
            x_val = 1000 * np.random.random_sample([1000, 100]).astype(x_type)
            for alpha in [0.1, -0.1, 1.0, -1.0]:
                def func(x):
                    x_ = tf.nn.leaky_relu(x, alpha)
                    return tf.identity(x_, name=_TFOUTPUT)
                self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("fails on caffe2 with dim issue")
    @check_onnxruntime_incompatibility("Mul")
    def test_leaky_relu_with_dependency(self):
        x_val = 1000 * np.random.random_sample([1000, 100]).astype(np.float32)
        def func(x):
            # simulate leaky_relu
            alpha = tf.constant(0.5)
            y = alpha * x
            x_ = tf.maximum(y, x)
            dependency = y - 1

            return tf.identity(x_, name=_TFOUTPUT), tf.identity(dependency, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val})

    @skip_caffe2_backend("fails on caffe2 with dim issue")
    @check_onnxruntime_incompatibility("Mul")
    def test_leaky_relu_float(self):
        x_val = 1000 * np.random.random_sample([1000, 100]).astype(np.float32)
        for alpha in [0.1, -0.1, 1.0, -1.0]:
            def func(x):
                x_ = tf.nn.leaky_relu(x, alpha)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Elu")
    def test_elu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.nn.elu(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Tanh")
    def test_tanh(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.tanh(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    def test_relu6(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0, 6, 7], dtype=np.float32).reshape((2, 3))
        def func(x):
            x_ = tf.nn.relu6(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Sub")
    def test_relu6_dynamic(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.nn.relu6(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_concat(self):
        x_val1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x_val2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        x_val3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.float32)
        def func(x1, x2, x3):
            x_ = tf.concat([x1, x2, x3], 0)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, "input3:0": x_val3})

    def test_concat_empty_const_input(self):
        x_val1 = np.array([1, 2, 3], dtype=np.float32)
        x_val2 = np.array([], dtype=np.float32)
        def func(x1):
            x2 = tf.constant(x_val2, dtype=tf.float32)
            x_ = tf.concat([x1, x2], 0)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1})

        x_val1 = np.array([[1, 2, 3]], dtype=np.float32)
        x_val2 = np.array([[]], dtype=np.float32)
        def func(x1):
            x2 = tf.constant(x_val2, dtype=tf.float32)
            x_ = tf.concat([x1, x2], 1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1})

        x_val1 = np.array([1, 2, 3], dtype=np.float32)
        x_val2 = np.array([], dtype=np.float32)
        x_val3 = np.array([13, 14, 15], dtype=np.float32)
        def func(x1, x3):
            x2 = tf.constant(x_val2, dtype=tf.float32)
            x_ = tf.concat([x1, x2, x3], 0)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val3})

    @check_opset_min_version(6, "cast")
    def test_concat_int64(self):
        x_val1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        x_val2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64)
        x_val3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.int64)
        def func(x1, x2, x3):
            x_ = tf.concat([x1, x2, x3], 0)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, "input3:0": x_val3})

    def test_concat_negative_axis(self):
        x_val1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x_val2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        x_val3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.float32)
        def func(x1, x2, x3):
            x_ = tf.concat([x1, x2, x3], -1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, "input3:0": x_val3})

    def test_concat_negative_axis_none_shape(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape((2, 3))
        y_val = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32).reshape((2, 3))
        s1_val = np.array([1, 1], dtype=np.int32)
        s2_val = np.array([1, 1], dtype=np.int32)
        def func():
            x = tf_placeholder(tf.float32, [2, 3], name=_TFINPUT)
            y = tf_placeholder(tf.float32, [2, 3], name=_TFINPUT1)
            s1 = tf_placeholder(tf.int32, [2], name="input3")
            s2 = tf_placeholder(tf.int32, [2], name="input4")
            s = tf.add(s1, s2)
            x_with_none_shape = tf.slice(x, [0, 0], s)
            t = tf.concat([x_with_none_shape, y], -1)
            return tf.identity(t, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, "input3:0": s1_val, "input4:0": s2_val},
                            as_session=True, premade_placeholders=True)

    def test_concat_const_string(self):
        x_val1 = np.array([["Hello world", "abc"], ["def", "♦♥♠♣"]], dtype=str)
        const_val = np.array([["Hello there", "wxyz"], ["", "π"]], dtype=str)
        def func(x1):
            x_ = tf.concat([x1, const_val], 0)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1})

    @check_onnxruntime_incompatibility("Pow")
    def test_pow(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
        e = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        def func(x):
            x_ = tf.pow(x, tf.constant(e))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_embedding_lookup(self):
        x_val1 = np.array([[1]], dtype=np.int32)
        x_val2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        def func(x):
            t = tf.constant(x_val2)
            x_ = tf.nn.embedding_lookup(t, x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1})

    @skip_tflite("Advanced constant shape folding not implemented for tflite")
    @skip_tfjs("Advanced constant folding not implemented for tfjs")
    def test_slice_from_shape_const_fold(self):
        x_val = np.array([4, 3], dtype=np.int64)
        x_shape = np.array([-1, 3], dtype=np.int64)
        def func(x):
            z = tf.zeros(x)
            x = tf.reshape(z, tf.constant(x_shape))
            s = tf.shape(x)
            t1 = tf.constant([1], dtype=tf.int32)
            t2 = tf.constant([2], dtype=tf.int32)
            y = tf.strided_slice(s, t1, t2, shrink_axis_mask=1)
            return tf.identity(y, name=_TFOUTPUT)
        def graph_validator(g):
            # After constant folding just an input and const output node remain
            return len(g.get_nodes()) == 2
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, graph_validator=graph_validator)

    @timeout(5)
    def test_slice_const_fold_halts(self):
        # Regression test for infinite loop during constant-folding.
        x_val = np.array([4, 3], dtype=np.int32)
        x_shape = np.array([-1, 3], dtype=np.int32)
        def func(x):
            x_reshaped = tf.reshape(tf.zeros(x), tf.constant(x_shape))
            s = tf.shape(x_reshaped)
            const1 = tf.constant([1], dtype=tf.int32)
            const2 = tf.constant([2], dtype=tf.int32)
            s_indexed = tf.strided_slice(s, const1, const2, strides=const1, shrink_axis_mask=1)
            x_indexed = tf.strided_slice(x, const1, const2, strides=const1, shrink_axis_mask=1)
            mul = tf.multiply(s_indexed, tf.constant(2, dtype=s_indexed.dtype))
            add = const1 + x_indexed + mul
            return tf.identity(add, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_slice(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        def func(x):
            t1 = tf.constant([0, 1], dtype=tf.int32)
            t2 = tf.constant([2, 2], dtype=tf.int32)
            x_ = tf.slice(x, t1, t2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_slice_neg_size(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        def func(x):
            t1 = tf.constant([0, 1], dtype=tf.int32)
            t2 = tf.constant([-1, 2], dtype=tf.int32)
            x_ = tf.slice(x, t1, t2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice in opset 10 can accept dymaic 'start' and 'ends'")
    def test_slice_with_non_const(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        t1 = np.array([0, 1], dtype=np.int32)
        t2 = np.array([2, 2], dtype=np.int32)
        def func(x, t1, t2):
            x_ = tf.slice(x, t1, t2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: t1, _INPUT2: t2})

    @check_opset_min_version(10, "Slice in opset 10 can accept dymaic 'start' and 'ends'")
    def test_slice_with_size_is_negative_one(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        t1 = np.array([0, 1], dtype=np.int32)
        # input "size" contains -1
        t2 = np.array([2, -1], dtype=np.int32)
        def func(x, t1, t2):
            x_ = tf.slice(x, t1, t2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: t1, _INPUT2: t2})

    @skip_caffe2_backend()
    def test_slice1(self):
        # FIXME: only 1 dimension supported by caffe2
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
        def func(x):
            t1 = tf.constant([1, 0, 0], dtype=tf.int32)
            t2 = tf.constant([1, 1, 3], dtype=tf.int32)
            x_ = tf.slice(x, t1, t2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_split(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape((5, 30))
        def func(x):
            x_, _, _ = tf.split(x, [4, 15, 11], 1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(13, "Split")
    def test_split_nonconst(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape((5, 30))
        y_val = np.array([4, 15, 11], np.int32)
        def func(x, y):
            x_, _, _ = tf.split(x, y, 1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @skip_tfjs("TFJS executes model incorrectly")
    def test_split_with_more_outputs(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape((5, 30))
        def func(x):
            return tf.split(x, [4, 15, 11], 1, name="split_test")
        self._run_test_case(func, ["split_test:0", "split_test:1", "split_test:2"], {_INPUT: x_val})

    def test_negative_split(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape((5, 30))
        def func(x):
            x_, _, _ = tf.split(x, [4, 15, -1], 1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_reducesum(self):
        # not supported by onnx-caffe2
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.reduce_sum(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(13, "ReduceSum")
    def test_reducesum_nonconst_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 1, 2))
        y_val = np.array([1, 2], dtype=np.int32)
        def func(x, y):
            x_ = tf.reduce_sum(x, axis=y)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(13, "ReduceSum")
    def test_reducesum_empty_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 1, 2))
        y_val = np.array([], dtype=np.int32)
        def func(x, y):
            x_ = tf.reduce_sum(x, axis=y)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(11, "ScatterND")
    def test_segment_sum_data_vector(self):
        segs_val = np.array([0, 0, 0, 1, 2, 2, 3, 3], dtype=np.int32)
        data_val = np.array([5, 1, 7, 2, 3, 4, 1, 3], dtype=np.float32)
        def func(data, segments):
            x_ = tf.math.segment_sum(data, segments)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: segs_val})

    @check_opset_min_version(11, "ScatterND")
    def test_segment_sum_unknown_rank(self):
        segs_val = np.array([0, 0, 0, 1, 2, 2, 3, 3], dtype=np.int32)
        data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
        data_shape_val = np.array([8, 2, 3, 1], dtype=np.int64)
        shape_pad_val = np.zeros((1, 2), dtype=np.int64)
        def func(data, segments, data_shape, shape_pad):
            # Some hackery to make the rank unknown
            data_shape_ = tf.pad(data_shape, shape_pad, constant_values=0)
            data = tf.reshape(data, data_shape_)
            x_ = tf.math.segment_sum(data, segments)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: data_val, _INPUT1: segs_val, _INPUT2: data_shape_val, _INPUT3: shape_pad_val})

    @check_opset_min_version(11, "ScatterND")
    def test_segment_ops_data_tensor(self):
        for tf_op in [tf.math.segment_sum, tf.math.segment_prod, tf.math.segment_min, tf.math.segment_max]:
            segs_val = np.array([0, 0, 0, 1, 2, 2, 3, 3], dtype=np.int32)
            data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
            def func(data, segments):
                x_ = tf_op(data, segments)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: segs_val})

    @check_opset_min_version(11, "ScatterND")
    @skip_tflite("unknown rank")
    def test_segment_mean_unknown_rank(self):
        segs_val = np.array([0, 0, 0, 1, 2, 2, 3, 3], dtype=np.int32)
        data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
        data_shape_val = np.array([8, 2, 3, 1], dtype=np.int64)
        shape_pad_val = np.zeros((1, 2), dtype=np.int64)
        def func(data, segments, data_shape, shape_pad):
            # Some hackery to make the rank unknown
            data_shape_ = tf.pad(data_shape, shape_pad, constant_values=0)
            data = tf.reshape(data, data_shape_)
            x_ = tf.math.segment_mean(data, segments)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: data_val, _INPUT1: segs_val, _INPUT2: data_shape_val, _INPUT3: shape_pad_val})

    @check_opset_min_version(11, "ScatterND")
    def test_sparse_segment_sum(self):
        data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
        indices_val = np.array([2, 0, 1, 3, 5, 4, 3, 5, 5], dtype=np.int32)
        segs_val = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3], dtype=np.int32)
        def func(data, indices, segments):
            x_ = tf.sparse.segment_sum(data, indices, segments)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: indices_val, _INPUT2: segs_val})

    @check_opset_min_version(11, "ScatterND")
    def test_sparse_segment_mean(self):
        data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
        indices_val = np.array([2, 0, 1, 3, 5, 4, 3, 5, 5], dtype=np.int32)
        segs_val = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3], dtype=np.int32)
        def func(data, indices, segments):
            x_ = tf.sparse.segment_mean(data, indices, segments)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: indices_val, _INPUT2: segs_val})

    @check_opset_min_version(11, "ScatterND")
    def test_sparse_segment_sqrtn(self):
        data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
        indices_val = np.array([2, 0, 1, 3, 5, 4, 3, 5, 5], dtype=np.int32)
        segs_val = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3], dtype=np.int32)
        def func(data, indices, segments):
            x_ = tf.sparse.segment_sqrt_n(data, indices, segments)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: indices_val, _INPUT2: segs_val})

    @check_opset_min_version(11, "ScatterND")
    def test_sparse_segment_ops_with_num_segments(self):
        for tf_op in [tf.sparse.segment_sum, tf.sparse.segment_mean, tf.sparse.segment_sqrt_n]:
            data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
            indices_val = np.array([2, 0, 1, 3, 5, 4, 3, 5, 5], dtype=np.int32)
            segs_val = np.array([0, 0, 0, 1, 3, 3, 4, 4, 4], dtype=np.int32)
            def func(data, indices, segments):
                x_ = tf_op(data, indices, segments, num_segments=6)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: indices_val, _INPUT2: segs_val})

    @check_opset_min_version(11, "ScatterND")
    @check_tf_min_version("2.3", "needs tf 2.3")
    def test_unsorted_segment_ops(self):
        tf_ops = [
            tf.math.unsorted_segment_max,
            tf.math.unsorted_segment_min,
            tf.math.unsorted_segment_sum,
            tf.math.unsorted_segment_prod,
            tf.math.unsorted_segment_mean,
            tf.math.unsorted_segment_sqrt_n,
        ]
        for tf_op in tf_ops:
            segs_val = np.array([1, 3, 0, 1, 2, 4, 2, 1], dtype=np.int32)
            data_val = np.arange(8 * 2 * 3, dtype=np.float32).reshape([8, 2, 3])
            def func(data, segments):
                x_ = tf_op(data, segments, num_segments=5)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: segs_val})

    @check_opset_min_version(11, "ScatterND")
    @check_tf_min_version("2.3", "num_segments can be int64 in tf 2.3")
    def test_segment_op_types(self):
        data_dtypes = [np.int32, np.float32]
        seg_dtypes = [np.int32, np.int64]
        for dtypes in product(data_dtypes, seg_dtypes, seg_dtypes, seg_dtypes):
            data_val = np.arange(8 * 2 * 3, dtype=dtypes[0]).reshape([8, 2, 3])
            indices_val = np.array([2, 0, 1, 3, 5, 4, 3, 5, 5], dtype=dtypes[1])
            segs_val = np.array([0, 0, 0, 1, 3, 3, 4, 4, 4], dtype=dtypes[2])
            def func(data, indices, segments):
                x_ = tf.sparse.segment_sum(data, indices, segments, num_segments=np.array(6, dtype=dtypes[3]))
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: data_val, _INPUT1: indices_val, _INPUT2: segs_val})

    @check_opset_min_version(11, "CumSum")
    @check_tf_min_version("1.14")
    def test_set_union(self):
        a_val = np.array([[10, 2, 30, 2, 5], [10, 9, 1, 9, 3]], np.int32)
        b_val = np.array([[4, 5, 10, 8, 9], [1, 4, 1, 1, 5]], np.int32)
        def func(a, b):
            s = tf.sets.union(a, b)
            indices, values, shape = s.indices, s.values, s.dense_shape
            indices = tf.identity(indices, name=_TFOUTPUT)
            values = tf.identity(values, name=_TFOUTPUT1)
            shape = tf.identity(shape, name=_TFOUTPUT2)
            return indices, values, shape
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: a_val, _INPUT1: b_val})

    @check_opset_min_version(11, "CumSum")
    @check_tf_min_version("1.14")
    def test_set_intersection(self):
        a_val = np.array([[10, 2, 30, 2, 5], [10, 9, 1, 9, 3]], np.int32)
        b_val = np.array([[4, 5, 10, 8, 9], [1, 4, 1, 1, 5]], np.int32)
        def func(a, b):
            s = tf.sets.intersection(a, b)
            indices, values, shape = s.indices, s.values, s.dense_shape
            indices = tf.identity(indices, name=_TFOUTPUT)
            values = tf.identity(values, name=_TFOUTPUT1)
            shape = tf.identity(shape, name=_TFOUTPUT2)
            return indices, values, shape
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: a_val, _INPUT1: b_val})

    @check_opset_min_version(11, "CumSum")
    @check_tf_min_version("1.14")
    def test_set_difference(self):
        a_val = np.array([[10, 2, 30, 2, 5], [10, 9, 1, 9, 3]], np.int32)
        b_val = np.array([[4, 5, 10, 8, 9], [1, 4, 1, 1, 5]], np.int32)
        for aminusb in [True, False]:
            def func(a, b):
                s = tf.sets.difference(a, b, aminusb)
                indices, values, shape = s.indices, s.values, s.dense_shape
                indices = tf.identity(indices, name=_TFOUTPUT)
                values = tf.identity(values, name=_TFOUTPUT1)
                shape = tf.identity(shape, name=_TFOUTPUT2)
                return indices, values, shape
            self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: a_val, _INPUT1: b_val})

    @check_onnxruntime_incompatibility("Sqrt")
    def test_sqrt(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.math.sqrt(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def _test_range_const(self, extra_opset=None):
        process_args = {}
        if extra_opset is not None:
            process_args["extra_opset"] = [extra_opset]

        def func():
            x = tf.range(5)
            return tf.identity(x, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)

        def func():
            x = tf.range(3, 3, 5)
            return tf.identity(x, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)

        def func():
            x = tf.range(0, -5, -2)
            return tf.identity(x, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)

        def func():
            x = tf.range(-5.0, 5.0, 1.5)
            return tf.identity(x, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)

        def func():
            x = tf.range(2.5, 5.0, 10.0)
            return tf.identity(x, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)

    def _test_range_non_const(self, extra_opset=None):
        process_args = {}
        if extra_opset is not None:
            process_args["extra_opset"] = [extra_opset]

        def func():
            x = tf.range(5.0)
            return tf.identity(x, name=_TFOUTPUT)
        g = self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)
        # TODO: tf-2.0 uses the optimizer which will most likely make the range const which is not what we want to test
        # self.assertTrue(extra_opset is None
        #                or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))

        def func():
            x = tf.range(0, -5.0, -2)
            return tf.identity(x*x, name=_TFOUTPUT)
        g = self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)
        # TODO: tf-2.0 uses the optimizer which will most likely make the range const  which is not what we want to test
        # self.assertTrue(extra_opset is None
        #                or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))

        # disable this case due to onnxruntime loop issue
        # https://github.com/microsoft/onnxruntime/issues/1272
        # x = tf.range(3.0, 3.0, 5)
        # return tf.identity(x, name=_TFOUTPUT)
        # g = self._run_test_case(func, [_OUTPUT], {}, process_args=process_args)
        # self.assertTrue(extra_opset is None
        #                 or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))

        delta_val = np.array(1.5, dtype=np.float32)
        def func(delta):
            x = tf.range(-5.0, 5.0, delta)
            return tf.identity(x, name=_TFOUTPUT)
        g = self._run_test_case(func, [_OUTPUT], {_INPUT: delta_val}, process_args=process_args)
        self.assertTrue(extra_opset is None
                        or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))

        start_val = np.array(2.5, dtype=np.float32)
        def func(start):
            x = tf.range(start, 5.0, 10.0)
            return tf.identity(x, name=_TFOUTPUT)
        g = self._run_test_case(func, [_OUTPUT], {_INPUT: start_val}, process_args=process_args)
        self.assertTrue(extra_opset is None
                        or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))

    @check_opset_min_version(7, "cast")
    def test_range_const(self):
        self._test_range_const()

    def test_range_non_const(self):
        self._test_range_non_const()

    @test_ms_domain()
    def test_ms_range_const(self, extra_opset):
        self._test_range_const(extra_opset)

    @test_ms_domain()
    def test_ms_range_non_const(self, extra_opset):
        self._test_range_non_const(extra_opset)

    @check_onnxruntime_incompatibility("Sqrt")
    def test_rsqrt(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.math.rsqrt(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_incompatibility("Reciprocal")
    def test_reciprocal(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.math.reciprocal(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04)

    def test_reducemax(self):
        # not supported by onnx-caffe2
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.reduce_max(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    def test_reducemax_global_max_pool(self):
        for keepdims in [True, False]:
            x_val = make_xval((2, 3, 4, 5, 6))
            def func(x):
                x_ = tf.reduce_max(x, axis=[2, 3, 4], keepdims=keepdims)
                return tf.add(x_, 0, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_reduceprod(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.reduce_prod(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_reducemean(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.reduce_mean(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_reducemean_global_avg_pool(self):
        for keepdims in [True, False]:
            x_val = make_xval((2, 3, 4, 5))
            def func(x):
                x_ = tf.reduce_mean(x, axis=[2, 3], keepdims=keepdims)
                return tf.add(x_, 0, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_onnxruntime_incompatibility("Pow")
    def test_pow_scalar(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
        e = np.array(2.0, dtype=np.float32)
        def func(x):
            x_ = tf.pow(x, tf.constant(e))
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_pad_const_default_val(self):
        params = [
            ("CONSTANT", [[1, 1], [2, 2]], [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]),
            ("CONSTANT", [[0, 0], [3, 3], [3, 3], [0, 0]], np.random.randn(1, 3, 4, 5).astype(np.float32)),
        ]
        for p in params:
            mode, pad, xv = p
            x_val = np.array(xv, dtype=np.float32)
            def func(x):
                paddings = tf.constant(pad)
                op = tf.pad(x, paddings, mode)
                return tf.identity(op, name=_TFOUTPUT)
            self.logger.debug(str(p))
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_pad_const(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        def func(x):
            paddings = tf.constant([[1, 1], [2, 2]], name="paddings")
            op = tf.pad(x, paddings, mode="CONSTANT", name="const_with_val", constant_values=999)
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_pad_reflect(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        def func(x):
            paddings = tf.constant([[1, 1], [2, 2]], name="paddings")
            op = tf.pad(x, paddings, mode="REFLECT", name="reflect")
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "Compress")
    def test_pad_symmetric(self):
        x_val = make_xval([4, 1, 5])
        def func(x):
            paddings = tf.constant([[1, 3], [0, 0], [2, 4]], name="paddings")
            op = tf.pad(x, paddings, mode="SYMMETRIC", name="symmetric")
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Pad")
    def test_dynamic_pad_symmetric(self):
        x_val = make_xval([4, 1, 5])
        y_val = np.array([[1, 3], [0, 0], [2, 4]], np.int32)
        def func(x, y):
            op = tf.pad(x, y, mode="SYMMETRIC", name="symmetric")
            return tf.identity(op, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @skip_caffe2_backend()
    def test_randomuniform(self):
        def func():
            shape = tf.constant([2, 3], name="shape")
            x_ = random_uniform(shape, name="rand", dtype=tf.float32)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case(func, [_OUTPUT], {}, check_value=False, check_shape=True)

    def test_random_std_normal(self):
        def func():
            shape = tf.constant([20, 10, 50], name="shape")
            x_ = tf.random.normal(shape)
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        g = self._run_test_case(func, [_OUTPUT], {}, check_value=False, check_shape=True)
        results = self.run_backend(g, g.outputs, {})[0]
        self.assertTrue(-0.1 < np.mean(results) < 0.1)
        self.assertTrue(0.9 < np.std(results) < 1.1)

    def test_randomnormal(self):
        def func():
            shape = tf.constant([20, 10, 50], name="shape")
            x_ = tf.random.normal(shape, mean=10, stddev=2)
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        g = self._run_test_case(func, [_OUTPUT], {}, check_value=False, check_shape=True)
        results = self.run_backend(g, g.outputs, {})[0]
        self.assertTrue(9.8 < np.mean(results) < 10.2)
        self.assertTrue(1.9 < np.std(results) < 2.1)

    @check_opset_min_version(9, "RandomNormalLike")
    def test_randomnormal_unknown_shape(self):
        shape_val = np.array([20, 10, 50], np.int32)
        def func(shape):
            x_ = tf.random.normal(shape)
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        feed_dict = {_INPUT: shape_val}
        g = self._run_test_case(func, [_OUTPUT], feed_dict, check_value=False, check_shape=True)
        if "input" in g.input_names:
            # TFLite inputs don't have port numbers
            feed_dict = {k.split(":")[0]: v for k, v in feed_dict.items()}
        results = self.run_backend(g, g.outputs, feed_dict)[0]
        self.assertTrue(-0.1 < np.mean(results) < 0.1)
        self.assertTrue(0.9 < np.std(results) < 1.1)

    @check_opset_min_version(10, "TopK")
    def test_random_shuffle(self):
        x_val = make_xval([5, 4, 3])
        def func(x):
            x_ = tf.random.shuffle(x)
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        g = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_value=False, check_shape=True)
        feed_dict = {_INPUT: x_val}
        if "input" in g.input_names:
            # TFLite inputs don't have port numbers
            feed_dict = {k.split(":")[0]: v for k, v in feed_dict.items()}
        results = self.run_backend(g, g.outputs, feed_dict)
        np.testing.assert_allclose(x_val, np.sort(results[0], axis=0))

    def test_randomuniform_int(self):
        def func():
            shape = tf.constant([100, 3], name="shape")
            x_ = random_uniform(shape, name="rand", dtype=tf.int32, minval=2, maxval=10)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        g = self._run_test_case(func, [_OUTPUT], {}, check_value=False, check_shape=True)
        results = self.run_backend(g, g.outputs, {})
        numbers = set(results[0].flatten())
        self.assertEqual(sorted(numbers), list(range(2, 10)))

    def test_randomuniform_int_scalar(self):
        def func():
            shape = tf.constant(np.array([], np.int32), name="shape")
            x_ = random_uniform(shape, name="rand", dtype=tf.int32, minval=2, maxval=10)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        g = self._run_test_case(func, [_OUTPUT], {}, check_value=False, check_shape=True)
        results = self.run_backend(g, g.outputs, {})
        self.assertTrue(2 <= results[0] < 10)

    def test_randomuniform_int_nonconst_max(self):
        m_val = np.array(8, dtype=np.int32)
        def func(m):
            shape = tf.constant([100, 3], name="shape")
            x_ = random_uniform(shape, name="rand", dtype=tf.int32, minval=0, maxval=m)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            return tf.identity(x_, name=_TFOUTPUT)
        g = self._run_test_case(func, [_OUTPUT], {_INPUT: m_val}, check_value=False, check_shape=True)
        feed_dict = {_INPUT: m_val}
        if "input" in g.input_names:
            # TFLite inputs don't have port numbers
            feed_dict = {k.split(":")[0]: v for k, v in feed_dict.items()}
        results = self.run_backend(g, g.outputs, feed_dict)
        numbers = set(results[0].flatten())
        self.assertEqual(sorted(numbers), list(range(8)))

    def test_randomuniform_int_nonconst_min_max(self):
        n_val = np.array(2, dtype=np.int32)
        m_val = np.array(10, dtype=np.int32)
        def func(n, m):
            shape = tf.constant([100, 3], name="shape")
            x_ = random_uniform(shape, name="rand", dtype=tf.int32, minval=n, maxval=m)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            return tf.identity(x_, name=_TFOUTPUT)
        g = self._run_test_case(func, [_OUTPUT], {_INPUT: n_val, _INPUT1: m_val}, check_value=False, check_shape=True)
        feed_dict = {_INPUT: n_val, _INPUT1: m_val}
        if "input" in g.input_names:
            # TFLite inputs don't have port numbers
            feed_dict = {k.split(":")[0]: v for k, v in feed_dict.items()}
        results = self.run_backend(g, g.outputs, feed_dict)
        numbers = set(results[0].flatten())
        self.assertEqual(sorted(numbers), list(range(2, 10)))

    @check_opset_min_version(9, "RandomUniformLike")
    def test_randomuniform_int_nonconst_min_max_shape(self):
        n_val = np.array(2, dtype=np.int32)
        m_val = np.array(10, dtype=np.int32)
        s_val = np.array([100, 3], dtype=np.int64)
        def func(n, m, s):
            x_ = random_uniform(s, name="rand", dtype=tf.int32, minval=n, maxval=m)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            return tf.identity(x_, name=_TFOUTPUT)
        g = self._run_test_case(func, [_OUTPUT], {_INPUT: n_val, _INPUT1: m_val, _INPUT2: s_val},
                                check_value=False, check_shape=True)
        feed_dict = {_INPUT: n_val, _INPUT1: m_val, _INPUT2: s_val}
        if "input" in g.input_names:
            # TFLite inputs don't have port numbers
            feed_dict = {k.split(":")[0]: v for k, v in feed_dict.items()}
        results = self.run_backend(g, g.outputs, feed_dict)
        numbers = set(results[0].flatten())
        self.assertEqual(sorted(numbers), list(range(2, 10)))

    @skip_caffe2_backend()
    @check_opset_after_tf_version("2.2", 9, "RandomUniform")
    def test_randomuniform_dyn_shape(self):
        # test for dynamic shape coming from a shape op
        x_val = np.array([0, 1, 2, 3, 5], dtype=np.int64)
        def func(x):
            ret = random_uniform(x[3:], dtype=tf.float32)
            return tf.identity(ret, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_value=False, check_shape=True)

    @skip_caffe2_backend()
    def test_randomuniform_calc_shape(self):
        # test for dynamic shape coming from some subgraph
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        def func(x):
            x_ = tf.identity(x)
            x_ = tf.shape(x_, name="shape")[1:]
            x_ = random_uniform(x_, name="rand", dtype=tf.float32)
            x_ = tf.identity(x_)
            return tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_value=False, check_shape=True)

    @check_opset_min_version(9, "Compress")
    @skip_onnx_checker("Checker fails type inference for Compress")
    def test_sample_distorted_bounding_box_v2(self):
        x_val = np.array([200, 300, 3], dtype=np.int32)
        y_val = np.random.uniform(size=[1, 1000, 4]).astype(np.float32)
        y_val = np.array([[0, 0, 0.1, 0.1], [0.9, 0.9, 1, 1]], np.float32).reshape([1, 2, 4])

        def func(image_size, bounding_boxes):
            begin, size, bboxes = tf.image.sample_distorted_bounding_box(
                image_size, bounding_boxes, seed=42, min_object_covered=0.8,
                aspect_ratio_range=[0.05, 3], area_range=[0.05, 1], max_attempts=100,
                use_image_if_no_bounding_boxes=False)
            begin_ = tf.identity(begin, name=_TFOUTPUT)
            size_ = tf.identity(size, name=_TFOUTPUT1)
            bboxes_ = tf.identity(bboxes, name=_TFOUTPUT2)
            return begin_, size_, bboxes_
        # since results are random, compare the shapes only
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: x_val, _INPUT1: y_val},
                            check_value=False, check_shape=True)

    @skip_caffe2_backend()
    def test_argminmax(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.argmin(x, axis=0)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array([1, 2, -2, -1], dtype=np.int32).reshape((2, 2))
        def func(x):
            x_ = tf.argmax(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array([1, 2, -2, -1], dtype=np.int32).reshape((2, 2))
        def func(x):
            x_ = tf.argmax(x, output_type=x_val.dtype)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(6, "cast")
    def test_cast(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.cast(x, tf.int32)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite does not support uint32 if tf version <= 2.3.0")
    @check_opset_min_version(6, "cast")
    def test_cast_unit32(self):
        x_val = np.array([1, 2, 3, 4], dtype=np.uint32).reshape((2, 2))
        def func(x):
            x_ = tf.cast(x, tf.uint64)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "sign")
    def test_sign(self):
        x_vals = [np.array([1.0, 2.0, 0.0, -1.0, 0.0, -2.0], dtype=np.float32).reshape((2, 3)),
                  np.array([1, 2, 0, -1, 0, -2], dtype=np.int32).reshape((2, 3)),
                  np.array([1, 2, 0, -1, 0, -2], dtype=np.int64).reshape((2, 3))]
        for x_val in x_vals:
            def func(x):
                x_ = tf.math.sign(x)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tfjs("tfjs produces incorrect results")
    def test_onehot0(self):
        x_val = np.array([0, 1, 2], dtype=np.int32)
        depth = 5
        for dtype, axis in [(tf.float32, -1), (tf.int64, 0), (tf.float64, 1)]:
            def func(x):
                val1 = tf.constant(5, dtype)
                val2 = tf.constant(1, dtype)
                x_ = tf.one_hot(x, depth, on_value=val1, axis=axis, off_value=val2, dtype=dtype)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @unittest.skip("only rank 1 is currently implemented")
    def test_onehot1(self):
        # only rank 1 is currently implemented
        x_val = np.array([[0, 2], [1, -1]], dtype=np.int32)
        depth = 3
        def func(x):
            x_ = tf.one_hot(x, depth, on_value=5.0, axis=-1, off_value=0.0, dtype=tf.float32)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_target("rs6", "onehot")
    def test_onehot2(self):
        for axis in [-1, 0, 1]:
            x_val = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 2], dtype=np.int32)
            depth = 20
            def func(x):
                x_ = tf.one_hot(x, depth, on_value=5.0, axis=axis, off_value=1.0, dtype=tf.float32)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_target("rs6", "onehot")
    @check_opset_min_version(9, "onehot")
    def test_onehot3(self):
        # rank 1
        for np_dtype in [np.int32, np.int64]:
            x_val = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 2], dtype=np_dtype)
            depth = np.array(20).astype(np.int64)
            def func(x):
                on_off = np.array([5.6, 1.2]).astype(np_dtype)
                x_ = tf.one_hot(x, depth, on_value=on_off[0], axis=-1, off_value=on_off[1])
                return tf.identity(x_, name=_TFOUTPUT)
            graph = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})
            self.assertTrue(len(group_nodes_by_type(graph)["OneHot"]) == 1, "onnx onehot should be used")
        # rank 2
        for aixs in [-1, 0, 1, 2]:
            for np_dtype in [np.int32, np.int64]:
                x_val = np.arange(0, 50, dtype=np_dtype).reshape([-1, 10])
                depth = np.array(20).astype(np.int64)
                def func(x):
                    on_off = np.array([5.6, 1.2]).astype(np_dtype)
                    x_ = tf.one_hot(x, depth, on_value=on_off[0], axis=aixs, off_value=on_off[1])
                    return tf.identity(x_, name=_TFOUTPUT)
                graph = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})
                self.assertTrue(len(group_nodes_by_type(graph)["OneHot"]) == 1, "onnx onehot should be used")

    @check_opset_min_version(9, "onehot")
    @skip_tfjs("tfjs produces incorrect results")
    def test_onehot_rank0(self):
        depth = 5
        for np_dtype in [np.int32, np.int64]:
            x_val = np.array(3, dtype=np_dtype)
            for axis in [-1, 0]:
                def func(x):
                    x_ = tf.one_hot(x, depth, axis=axis)
                    return tf.identity(x_, name=_TFOUTPUT)
                self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("issue undefined dim 1")
    @check_tf_max_version("1.15", "not supported in tf-2.0")
    def test_flatten0(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        def func(x):
            x_ = tf.contrib.layers.flatten(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("issue undefined dim 1")
    @check_tf_max_version("1.15", "not supported in tf-2.0")
    def test_flatten1(self):
        x_val = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
        def func(x):
            x_ = tf.contrib.layers.flatten(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_max_version("1.15", "not supported in tf-2.0")
    def test_flatten2(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        def func(x):
            x_ = tf.contrib.layers.flatten(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_cancel_transpose(self):
        x_val = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
        def func(x):
            x_ = tf.identity(x, _TFINPUT)
            x_ = tf.transpose(x_, perm=NHWC_TO_NCHW)
            x_ = tf.transpose(x_, perm=NCHW_TO_NHWC)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_min_version("0.5.0", "topk-10's shape inference function has a bug")
    @check_opset_min_version(6, "cast")
    def test_topk1(self):
        x_val = np.arange(3 * 2 * 3).astype("float32")
        def func(x):
            values, _ = tf.nn.top_k(x, 5, sorted=True)
            return tf.identity(values, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "TopK with dynamic K")
    def test_topk2(self):
        x_val = np.arange(3 * 2 * 3).astype("float32")
        k_val = np.array(10).astype(np.int32)
        def func(x, k):
            values, _ = tf.nn.top_k(x, k, sorted=True)
            return tf.identity(values, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: k_val})

    @check_onnxruntime_min_version("0.5.0", "topk-10's shape inference function has a bug")
    def test_topk3(self):
        # test topk index output
        x_val = np.arange(3 * 2 * 3).astype("float32")
        def func(x):
            _, idx = tf.nn.top_k(x, 5, sorted=True)
            return tf.identity(idx, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_stack_axis(self):
        for axis in [0, 1]:
            x_val = [np.random.randn(3, 4).astype("float32") for _ in range(10)]
            def func():
                x = [tf.constant(x_val[i], dtype=tf.float32) for i in range(10)]
                x_ = tf.stack(x, axis=axis)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {})

    def test_unstack_axis(self):
        for axis in [0, 1]:
            x_val = np.random.randn(10, 3, 4).astype("float32")
            def func():
                x = tf.constant(x_val, dtype=tf.float32)
                x_ = tf.unstack(x, axis=axis)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {})

    def _test_reorganize_data(self, op, shape):
        x_val = make_xval(shape)
        def func(x):
            x_ = op(x, block_size=2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("Space2Depth not implemented")
    def test_space_to_depth(self):
        self._test_reorganize_data(tf.nn.space_to_depth, [1, 28, 28, 3])

    @skip_caffe2_backend("Depth2Space not implemented")
    def test_depth_to_space(self):
        self._test_reorganize_data(tf.nn.depth_to_space, [1, 14, 14, 12])

    def _test_reorganize_data_gpu(self, op, shape):
        x_val = make_xval(shape)
        def func(x):
            x_ = op(x, block_size=2, data_format="NCHW")
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tf_cpu("only tf_gpu can run Space2Depth with NCHW format")
    @skip_caffe2_backend("Space2Depth not implemented")
    def test_space_to_depth_gpu(self):
        self._test_reorganize_data_gpu(tf.nn.space_to_depth, [1, 3, 50, 80])

    @skip_tf_cpu("only tf_gpu can run Depth2Space with NCHW format")
    @skip_caffe2_backend("Depth2Space not implemented")
    def test_depth_to_space_gpu(self):
        self._test_reorganize_data_gpu(tf.nn.depth_to_space, [1, 120, 25, 40])

    @check_opset_min_version(6, "addn")
    def test_addn(self):
        x_val = np.arange(3 * 2 * 3).astype("float32")
        def func(x):
            x_ = tf.add_n([x, x, x])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice1(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        def func(x):
            x_ = tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_strided_slice2(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        def func(x):
            x_ = tf.strided_slice(x, [1, 0, 0], [2, 2, 3], [1, 1, 1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_strided_slice3(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        def func(x):
            x_ = x[1:]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_strided_slice4(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        def func(x):
            x_ = x[:2]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice5(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        def func(x):
            x_ = x[:2, 0:1, 1:]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice6(self):
        # example from here:
        # https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/ops/strided-slice
        x_val = np.arange(5 * 6).astype("float32").reshape((5, 6))
        def func(x):
            x_ = x[2, :]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice7(self):
        x_val = np.arange(5 * 6).astype("float32").reshape((5, 6))
        def func(x):
            x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], begin_mask=2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        def func(x):
            x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], end_mask=2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        def func(x):
            x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], shrink_axis_mask=2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        def func(x):
            x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], ellipsis_mask=2)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice8(self):
        x_val = np.arange(1 * 2 * 3 * 4 * 5 * 6).astype("float32").reshape((1, 2, 3, 4, 5, 6))
        def func(x):
            x_ = x[0:1, ..., 1, 2:, :6]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.arange(1 * 2 * 3 * 4 * 5 * 6).astype("float32").reshape((1, 2, 3, 4, 5, 6))
        def func(x):
            x_ = x[0:1, 1, 2:, :6, ...]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.arange(1 * 2 * 3 * 4 * 5 * 6).astype("float32").reshape((1, 2, 3, 4, 5, 6))
        def func(x):
            x_ = x[..., 0:1, 1, 2:, :6]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice_dynamic_1(self):
        # simple case
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        y_val = np.array([0, 1, 2], dtype=np.int32)
        def func(x, y):
            x_ = tf.strided_slice(x, y, [2, 2, 3], [1, 1, 1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(10, "Slice")
    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice_dynamic_2(self):
        # int32
        x_val = np.arange(3 * 2 * 3).astype("int32").reshape((3, 2, 3))
        y_val = np.array([0, 1, 2], dtype=np.int32)
        def func(x, y):
            x_ = tf.strided_slice(x, y, [2, 2, 3], [1, 1, 1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(10, "Slice")
    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice_dynamic_3(self):
        # common usage, ellipsis_mask
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        y_val = np.array(1, dtype=np.int32)
        def func(x, y):
            x_ = x[y:2, :, :]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @skip_tflite("tflite converts strided slice incorrectly (steps 1 dim larger than starts/stops)")
    @check_opset_min_version(10, "Slice")
    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice_dynamic_4(self):
        # begin_mask, end_mask
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        y_val = np.array(1, dtype=np.int32)
        def func(x, y):
            x_ = x[y:, :y]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @skip_tflite("tflite converts strided slice incorrectly (steps 1 dim larger than starts/stops)")
    @check_opset_min_version(10, "Slice")
    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice_dynamic_5(self):
        # only slice the first axis
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        y_val = np.array(1, dtype=np.int32)
        def func(x, y):
            x_ = x[y:2]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @skip_tflite("tflite converts strided slice incorrectly (steps 1 dim larger than starts/stops)")
    @check_opset_min_version(10, "Slice")
    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice_dynamic_6(self):
        # shrink mask
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        y_val = np.array(1, dtype=np.int32)
        def func(x, y):
            x_ = x[y]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        y_val = np.array(-1, dtype=np.int32)
        def func(x, y):
            x_ = x[y]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(10, "Slice")
    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice_dynamic_7(self):
        x_val = np.arange(1 * 2 * 3 * 4 * 5 * 6).astype("float32").reshape((1, 2, 3, 4, 5, 6))
        y_val = np.array(1, dtype=np.int32)
        def func(x, y):
            x_ = x[0:y, ..., y, y:, :y]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

        x_val = np.arange(1 * 2 * 3 * 4 * 5 * 6).astype("float32").reshape((1, 2, 3, 4, 5, 6))
        y_val = np.array(1, dtype=np.int32)
        def func(x, y):
            x_ = x[0:y, y, y:, :y, ...]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

        x_val = np.arange(1 * 2 * 3 * 4 * 5 * 6).astype("float32").reshape((1, 2, 3, 4, 5, 6))
        y_val = np.array(1, dtype=np.int32)
        def func(x, y):
            x_ = x[..., 0:y, y, y:, :y]
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(10, "Slice")
    def test_strided_slice_reverse_1(self):
        x_val = np.arange(16 * 32).astype(np.float32).reshape((1, 16, 32, 1))
        def func(x):
            return tf.concat([x[:, :, :10], x[:, :, :21:-1]], axis=0, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_strided_slice_reverse_2(self):
        x_val = np.arange(16 * 32).astype(np.float32).reshape((1, 16, 32, 1))
        def func(x):
            return tf.concat([x[:, :, :10], x[:, :, 9::-1]], axis=0, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite converts strided slice incorrectly (steps 1 dim larger than starts/stops)")
    @check_opset_min_version(10, "Slice")
    def test_strided_slice_reverse_3(self):
        x_val = np.zeros((1, 16, 32, 1)).astype(np.float32)
        y_val = np.array(9).astype(np.int32)
        z_val = np.array(-1).astype(np.int32)
        def func(x, y, z):
            return tf.concat([x[:, :, :10], x[:, :, y::z]], axis=0, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_opset_min_version(10, "Slice")
    @skip_tfjs("TFJS executes model incorrectly")
    def test_new_axis_mask(self):
        def func(x, y):
            x_ = x[tf.newaxis, 0:y, y::2, tf.newaxis, :, tf.newaxis, :y, tf.newaxis, ..., 9]
            return tf.identity(x_, name=_TFOUTPUT)
        x_val = np.arange(5*10*10*10*10*20*30).astype("float32").reshape((5, 10, 10, 10, 10, 20, 30))
        y_val = np.array(9, dtype=np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(10, "Slice")
    @skip_tflite("not supported in tflite")
    def test_strided_slice_ellipse(self):
        def func1(x):
            x_ = x[..., tf.newaxis]
            return tf.identity(x_, name=_TFOUTPUT)
        shape = [1, 8, 64]
        x_val = np.arange(np.prod(shape)).astype("float32").reshape(shape)
        self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val})

        def func2(x):
            x_ = x[:, tf.newaxis, ..., :, tf.newaxis]
            return tf.identity(x_, name=_TFOUTPUT)
        shape = [2, 3, 4, 5]
        x_val = np.arange(np.prod(shape)).astype("float32").reshape(shape)
        self._run_test_case(func2, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    @skip_tflite("not supported in tflite")
    def test_strided_slice_only_ellipsis(self):
        def func1(x):
            x_ = x[...]
            return tf.identity(x_, name=_TFOUTPUT)
        shape = [1, 8, 64]
        x_val = np.arange(np.prod(shape)).astype("float32").reshape(shape)
        self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "batchnorm")
    def test_fused_batchnorm(self):
        x_shape = [1, 28, 28, 2]
        x_dtype = np.float32
        scale_dtype = np.float32
        scale_shape = [2]
        # only nhwc is support on cpu for tensorflow
        data_format = "NHWC"
        x_val = np.random.random_sample(x_shape).astype(x_dtype)
        scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        var_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        def func(x):
            scale = tf.constant(scale_val, name='scale')
            offset = tf.constant(offset_val, name='offset')
            mean = tf.constant(mean_val, name='mean')
            var = tf.constant(var_val, name='variance')
            epsilon = 0.001
            y, _, _ = fused_batch_norm(
                x, scale, offset, mean=mean, variance=var,
                epsilon=epsilon, data_format=data_format, is_training=False)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04)

    @check_opset_min_version(7, "batchnorm")
    @check_tf_min_version("2.4", "tf version above 2.4 supports NDHWC")
    def test_fused_batchnorm_3d(self):
        x_shape = [1, 28, 28, 2, 2]
        x_dtype = np.float32
        scale_dtype = np.float32
        scale_shape = [2]
        data_format = "NDHWC"
        x_val = np.random.random_sample(x_shape).astype(x_dtype)
        scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        var_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        def func(x):
            scale = tf.constant(scale_val, name='scale')
            offset = tf.constant(offset_val, name='offset')
            mean = tf.constant(mean_val, name='mean')
            var = tf.constant(var_val, name='variance')
            epsilon = 0.001
            y, _, _ = fused_batch_norm(
                x, scale, offset, mean=mean, variance=var,
                epsilon=epsilon, data_format=data_format, is_training=False)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04)

    @check_opset_min_version(7, "batchnorm")
    @skip_tfjs("TFJS executes model incorrectly")
    def test_fused_batchnorm_training(self):
        x_shape = [1, 28, 28, 2]
        x_dtype = np.float32
        scale_dtype = np.float32
        scale_shape = [2]
        # only nhwc is support on cpu for tensorflow
        data_format = "NHWC"
        x_val = np.random.random_sample(x_shape).astype(x_dtype)
        scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        def func(x):
            scale = tf.constant(scale_val, name='scale')
            offset = tf.constant(offset_val, name='offset')
            epsilon = 0.001
            y, _, _ = fused_batch_norm(
                x, scale, offset, mean=None, variance=None,
                epsilon=epsilon, data_format=data_format, is_training=True)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-04)

    @skip_tflite("tflite converts aborts")
    @skip_tfjs("TFJS executes model incorrectly")
    @check_opset_min_version(11, "batchnorm")
    @check_tf_min_version("2.4")
    def test_batchnorm_mixed(self):
        x_shape = [1, 32, 32, 2]
        x_dtype = np.float16
        scale_dtype = np.float32
        scale_shape = [2]
        x_val = np.random.random_sample(x_shape).astype(x_dtype)
        scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        var_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        def func(x, mean, offset, var):
            scale = tf.constant(scale_val, name='scale')
            y = tf.raw_ops.FusedBatchNormV3(x=x, scale=scale, offset=offset, mean=mean, variance=var,
                                            is_training=False, name=_TFOUTPUT)
            return y
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: x_val, _INPUT1: mean_val, _INPUT2: offset_val, _INPUT3: var_val})

    @check_opset_min_version(7, "batchnorm")
    @check_tf_min_version("1.13")
    def test_batchnorm(self):
        x_shape = [1, 128, 128, 2]
        x_dtype = np.float32
        scale_dtype = np.float32
        scale_shape = [2]
        x_val = np.random.random_sample(x_shape).astype(x_dtype)
        scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        var_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        def func(x, mean, offset, var):
            scale = tf.constant(scale_val, name='scale')
            epsilon = 0.001
            y = tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: mean_val, _INPUT2: offset_val, _INPUT3: var_val})

    @check_opset_min_version(7, "batchnorm")
    def test_conv2d_batchnorm_fusion(self):
        x_shape = [1, 28, 28, 2]
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape(_KERNEL3x3)
        # 2 channels for input and output
        w = np.concatenate([w, w, w, w]).reshape([3, 3, 2, 2])
        scale_dtype = np.float32
        scale_shape = x_shape[-1:]
        scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        var_val = np.random.random_sample(scale_shape).astype(scale_dtype)

        def func_conv2d(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
            return conv

        def func_fusedbn(x):
            scale = tf.constant(scale_val, name='scale')
            offset = tf.constant(offset_val, name='offset')
            mean = tf.constant(mean_val, name='mean')
            var = tf.constant(var_val, name='variance')
            epsilon = 0.1234
            y, _, _ = fused_batch_norm(
                func_conv2d(x), scale, offset, mean=mean, variance=var,
                epsilon=epsilon, data_format='NHWC', is_training=False)
            return tf.identity(y, name=_TFOUTPUT)

        def graph_validator(g):
            if 'BatchNormalization' in [n.type for n in g.get_nodes()]:
                return False
            return True

        self._run_test_case(func_fusedbn, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05, graph_validator=graph_validator)

    @check_opset_min_version(7, "batchnorm")
    def test_multiple_conv2d_fused_batchnorm(self):
        x_shape = [1, 28, 28, 2]
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape(_KERNEL3x3)
        # 2 channels for input and output
        w = np.concatenate([w, w, w, w]).reshape([3, 3, 2, 2])
        scale_dtype = np.float32
        scale_shape = x_shape[-1:]
        scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
        var_val = np.random.random_sample(scale_shape).astype(scale_dtype)

        def func_conv2d(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
            return conv

        def func_multiple_fusedbn(x):
            scale = tf.constant(scale_val, name='scale')
            offset = tf.constant(offset_val, name='offset')
            mean = tf.constant(mean_val, name='mean')
            var = tf.constant(var_val, name='variance')
            epsilon = 0.1234
            y, _, _ = fused_batch_norm(
                func_conv2d(x), scale, offset, mean=mean, variance=var,
                epsilon=epsilon, data_format='NHWC', is_training=False)

            y = tf.nn.relu(y)

            y, _, _ = fused_batch_norm(
                func_conv2d(y), scale, offset, mean=mean, variance=var,
                epsilon=epsilon, data_format='NHWC', is_training=False)

            y, _, _ = fused_batch_norm(
                func_conv2d(y), scale, offset, mean=mean, variance=var,
                epsilon=epsilon, data_format='NHWC', is_training=False)

            return tf.identity(y, name=_TFOUTPUT)

        def graph_validator(g):
            if 'BatchNormalization' in [n.type for n in g.get_nodes()]:
                return False
            return True

        self._run_test_case(func_multiple_fusedbn, [_OUTPUT], {_INPUT: x_val}, rtol=1e-05,
                            graph_validator=graph_validator)

    @check_tf_min_version("1.15")
    @check_opset_min_version(10, "quantize_and_dequantize")
    def test_qdq_unsigned_input(self):
        x_shape = [3, 3, 2]
        x_val = np.arange(1, 1+np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x):
            x_ = quantize_and_dequantize(x, 1.0, 6.0, signed_input=False, range_given=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.15")
    @check_opset_min_version(10, "quantize_and_dequantize")
    def test_qdq_optimizer(self):
        x_shape = [3, 3, 2]
        x_val = np.arange(1, 1+np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x):
            x_ = quantize_and_dequantize(x, 1.0, 6.0, signed_input=False, range_given=True)
            x_ = tf.transpose(x_, [1, 2, 0])
            x_ = tf.reshape(x_, tf.constant([9, 2]))
            x_ = quantize_and_dequantize(x_, 1.0, 6.0, signed_input=False, range_given=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val},
                                graph_validator=lambda g: check_op_count(g, "DequantizeLinear", 1, disabled=False))

    @check_tf_min_version("1.15")
    @check_opset_min_version(10, "quantize_and_dequantize")
    def test_qdq_optimizer_split_concat(self):
        x_shape = [7, 3, 5]
        y_shape = [7, 2, 5]
        x_val = np.arange(1, 1+np.prod(x_shape)).astype("float32").reshape(x_shape)
        y_val = np.arange(1, 1+np.prod(y_shape)).astype("float32").reshape(y_shape)
        def func(x, y):
            x_ = quantize_and_dequantize(x, 1.0, 30.0, signed_input=False, range_given=True)
            a, _, c = tf.unstack(x_, axis=1)
            ac = tf.stack([a, c], axis=1)
            y_ = quantize_and_dequantize(y, 1.0, 30.0, signed_input=False, range_given=True)
            m = tf.matmul(ac, tf.transpose(y_, [0, 2, 1]))
            m_ = m[2:, :, :]
            m_ = quantize_and_dequantize(m_, 1.0, 30.0, signed_input=False, range_given=True)
            return tf.identity(m_, name=_TFOUTPUT)
        def validate_graph(g):
            # MatMul should be wrapped in Dq/Q
            for n in g.get_nodes():
                if n.type == "MatMul":
                    if not all(inp.type == "DequantizeLinear" for inp in n.inputs):
                        return False
                    if not all(c.type == "QuantizeLinear" for c in g.find_output_consumers(n.output[0])):
                        return False
            return True

        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val}, graph_validator=validate_graph)

    @check_tf_min_version("1.15")
    @check_opset_min_version(11, "ScatterND")
    @skip_tflite("TFLite uses a pattern for ScatterND so number of DequantizeLinear won't match")
    def test_qdq_optimizer_scatter(self):
        x_val = np.array([10, 20, 30, 40], dtype=np.float32).reshape((4))
        y_val = np.array([0, 2], dtype=np.int64).reshape((2, 1))
        z_val = np.array([8, 11], dtype=np.float32).reshape((2))

        def func(x, y, z):
            x_ = quantize_and_dequantize(x, 1.0, 30.0, signed_input=False, range_given=True)
            z_ = quantize_and_dequantize(z, 1.0, 30.0, signed_input=False, range_given=True)
            w = tf.tensor_scatter_nd_update(x_, y, z_)
            w_ = quantize_and_dequantize(w, 1.0, 30.0, signed_input=False, range_given=True)
            return tf.identity(w_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val},
                            graph_validator=lambda g: check_op_count(g, "DequantizeLinear", 1, disabled=False))

        def func(x, y, z):
            x_ = quantize_and_dequantize(x, 1.0, 30.0, signed_input=False, range_given=True)
            w = tf.tensor_scatter_nd_update(x_, y, z)
            w_ = quantize_and_dequantize(w, 1.0, 30.0, signed_input=False, range_given=True)
            return tf.identity(w_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_tf_min_version("1.15")
    @check_opset_min_version(10, "quantize_and_dequantize")
    def test_qdq_dyn_range_unsigned_input(self):
        x_shape = [3, 3, 2]
        x_val = np.arange(1, 1+np.prod(x_shape)).astype("float32").reshape(x_shape) + 0.1
        def func(x):
            x_ = quantize_and_dequantize(x, 1.0, 6.0, signed_input=False, range_given=False)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite converter mistranslates quantize op")
    @check_tf_min_version("1.15")
    @check_opset_min_version(10, "quantize_and_dequantize")
    def test_qdq_signed_input(self):
        x_shape = [3, 3, 2]
        x_val = np.arange(-np.prod(x_shape)/2, np.prod(x_shape)/2).astype("float32").reshape(x_shape)
        def func(x):
            x_ = quantize_and_dequantize(x, -6.0, 6.0, signed_input=True, narrow_range=False, range_given=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite converter crashes")
    @check_tf_min_version("2.0")
    @check_opset_min_version(13, "quantize_and_dequantize")
    def test_qdq_per_channel_signed_input(self):
        x_shape = [3, 3, 2]
        x_val = np.arange(-np.prod(x_shape)/2, np.prod(x_shape)/2).astype("float32").reshape(x_shape)
        def func(x):
            x_ = quantize_and_dequantize(x, np.array([-1.72, -3.89]).astype(np.float32), \
                                         np.array([5.12, 2.36]).astype(np.float32), \
                                         signed_input=True, narrow_range=False, \
                                         range_given=True, axis=-1)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite converter crashes")
    @check_tf_min_version("2.0")
    @check_opset_min_version(13, "quantize_and_dequantize")
    def test_qdq_dyn_range_per_channel_signed_input(self):
        x_shape = [3, 3, 2]
        x_val = np.arange(-np.prod(x_shape)/2, np.prod(x_shape)/2).astype("float32").reshape(x_shape)
        def func(x):
            x_ = quantize_and_dequantize(x, np.array([-1.72, -3.89]).astype(np.float32), \
                                         np.array([5.12, 2.36]).astype(np.float32), \
                                         signed_input=True, narrow_range=False, \
                                         range_given=False, axis=-1)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_opset_min_version(7, "resize_nearest_neighbor")
    def test_resize_nearest_neighbor(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_nearest_neighbor(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "resize_nearest_neighbor")
    def test_resize_nearest_neighbor_with_non_const(self):
        x_shape = [3, 10, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1, dtype=np.float32).reshape(x_shape)
        x_new_size = np.array([20, 16]).astype(np.int32)
        def func(x, x_new_size_):
            x_ = resize_nearest_neighbor(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @skip_caffe2_backend()
    def test_resize_area(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_area(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    # https://github.com/microsoft/onnxruntime/issues/17564
    @skip_onnxruntime_backend("Blocked by onnxruntime issue #17564")
    @skip_caffe2_backend()
    def test_resize_area_align_coreners(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_area(x, x_new_size_, align_corners=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})


    @skip_caffe2_backend()
    @check_opset_min_version(7, "resize_bilinear")
    def test_resize_bilinear(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_bilinear(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "coordinate_transformation_mode attr of resize_bilinear")
    def test_resize_bilinear_align_coreners(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_bilinear(x, x_new_size_, align_corners=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "coordinate_transformation_mode attr")
    def test_resize_bilinear_half_pixel_centers(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_bilinear(x, x_new_size_, half_pixel_centers=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "resize_bilinear")
    def test_resize_bilinear_with_non_const(self):
        x_shape = [3, 10, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1, dtype=np.float32).reshape(x_shape)
        x_new_size = np.array([20, 16]).astype(np.int32)
        def func(x, x_new_size_):
            x_ = resize_bilinear(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @check_opset_min_version(10, "resize scale can less than 1")
    def test_resize_bilinear_with_non_const2(self):
        # scales has an element larger than 1 and also has an element less that 1
        x_shape = [3, 100, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1, dtype=np.float32).reshape(x_shape)
        x_new_size = np.array([20, 16]).astype(np.int32)
        def func(x, x_new_size_):
            x_ = resize_bilinear(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "resize_bilinear_v2")
    def test_resize_bilinear_v2_with_non_const(self):
        x_shape = [3, 10, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1, dtype=np.float32).reshape(x_shape)
        x_new_size = np.array([20, 16]).astype(np.int32)
        def func(x, x_new_size_):
            x_ = resize_bilinear_v2(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    def test_adjust_contrast(self):
        def func(x, y):
            x_ = tf.image.adjust_contrast(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        for x_shape in [[4, 3, 2], [2, 3, 4, 5], [3, 4, 2, 4, 3]]:
            x_val = np.arange(1, 1 + np.prod(x_shape), dtype=np.float32).reshape(x_shape)
            y_val = np.array(2.1, np.float32)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(11, "GatherElements")
    def test_adjust_saturation(self):
        x_val = np.array([[1, 2, 3], [4, 4, 4], [3, 2, 3], [3, 2, 2]], dtype=np.float32).reshape([2, 2, 3])
        y_val = np.array(2.1, np.float32)
        def func(x, y):
            x_ = tf.image.adjust_saturation(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})
        y_val = np.array(0.5, np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(11, "GatherND")
    def test_adjust_hue(self):
        x_val = np.array([[1, 2, 3], [4, 4, 4], [10, 2, 1], [10, 1, 2],
                          [4, 6, 5], [5, 6, 4], [1, 3, 2], [3, 5, 3]], dtype=np.float32).reshape([2, 4, 3])
        def func(x, y):
            x_ = tf.image.adjust_hue(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        for i in range(-10, 10, 2):
            y_val = np.array(i / 10, np.float32)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val}, rtol=1e-6, atol=2e-5)

    # https://github.com/microsoft/onnxruntime/issues/12302
    @skip_onnxruntime_backend("resize op can't work well under Cubic mode with ORT 1.12")
    @check_tf_min_version("2.0", "Results are slightly different in tf1")
    @check_opset_min_version(11, "resize bicubic")
    def test_resize_bicubic(self):
        x_shape = [1, 15, 20, 2]
        new_size_val = np.array([30, 40], dtype=np.int32)
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1).astype("float32").reshape(x_shape)
        def func(x, new_size):
            y = tf.image.resize(x, new_size, method=tf.image.ResizeMethod.BICUBIC)
            return tf.identity(y, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: new_size_val}, rtol=1e-6, atol=1e-5)

    @check_opset_min_version(10, "resize scale can less than 1")
    def test_resize_nearest_neighbor2(self):
        x_shape = [1, 300, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_nearest_neighbor(x, x_new_size_)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "coordinate_transformation_mode attr")
    def test_resize_nearest_neighbor_half_pixel_centers(self):
        x_shape = [1, 10, 20, 2]
        x_new_size = [20, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_nearest_neighbor(x, x_new_size_, half_pixel_centers=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "coordinate_transformation_mode and nearest_mode attr")
    def test_resize_nearest_neighbor_align_corners(self):
        x_shape = [1, 10, 20, 2]
        x_new_size = [20, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)/10, 0.1).astype("float32").reshape(x_shape)
        def func(x):
            x_new_size_ = tf.constant(x_new_size)
            x_ = resize_nearest_neighbor(x, x_new_size_, align_corners=True)
            return tf.identity(x_, name=_TFOUTPUT)
        _ = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "fill")
    def test_fill_float32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x0):
            x1 = tf.fill(x_val.shape, 9.0)
            x2 = tf.add(x0, x1)
            return tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "fill")
    def test_fill_int32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("int32").reshape(x_shape)
        def func(x0):
            x1 = tf.fill(x_val.shape, 9)
            x2 = tf.add(x0, x1)
            return tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "fill")
    def test_fill7_float32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        def func(x0):
            x1 = tf.fill(x_val.shape, 9.0)
            x2 = tf.add(x0, x1)
            return tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "fill")
    def test_fill7_int32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("int32").reshape(x_shape)
        def func(x0):
            x1 = tf.fill(x_val.shape, 9)
            x2 = tf.add(x0, x1)
            return tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "div")
    def test_tf_div(self):
        # pylint: disable=E0001,C0415
        from tensorflow.python.ops.gen_math_ops import div
        shape = 1000
        # test floating data
        x_val = (np.random.sample(shape) + 1e-6).astype(np.float32)
        y_val = (np.random.sample(shape) + 1e-6).astype(np.float32)
        def func(x, y):
            output = div(x, y, name=_TFOUTPUT)
            # assert output.op.type == "Div"
            return output
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

        # test integer data
        x_val = (100 * np.random.sample(shape) + 1).astype(np.int32)
        y_val = (100 * np.random.sample(shape) + 1).astype(np.int32)
        def func(x, y):
            output = div(x, y, name=_TFOUTPUT)
            # assert output.op.type == "Div"
            return output
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(7, "erf")
    def test_erf(self):
        x_shape = [2, 2]
        x_val0 = np.random.random(np.prod(x_shape)).astype(np.float32).reshape(x_shape)
        x_val1 = np.array([[-1, -0.5], [1, 0.5]]).astype(np.float32)
        for x_val in [x_val0, x_val1]:
            def func(x):
                x_ = tf.math.erf(x)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=0.01)

    @check_opset_min_version(8, "Scan")
    @skip_opset(9, "ReverseSequence")
    def test_reverse_sequence_batch_major(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [0, 0, 0], [0, 0, 0]]],
                         dtype=np.float32)
        def func(x):
            x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[2, 3, 1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3],
                          [4, 5, 6], [4, 5, 6], [1, 1, 1],
                          [0, 0, 0], [7, 8, 9], [0, 0, 0]
                          ],
                         dtype=np.float32)
        def func(x):
            x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[3] * 9)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val_shape = [5, 5, 7, 8, 9]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[5, 5, 5, 5, 5])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "Scan")
    @skip_opset(9, "ReverseSequence")
    def test_reverse_sequence_time_major(self):
        x_val = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                          [[4, 5, 6], [4, 5, 6], [0, 0, 0]],
                          [[0, 0, 0], [7, 8, 9], [0, 0, 0]]],
                         dtype=np.float32)
        def func(x):
            x_ = tf.reverse_sequence(x, seq_axis=0, batch_axis=1, seq_lengths=[2, 3, 1])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})
        x_val = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3],
                          [4, 5, 6], [4, 5, 6], [1, 1, 1],
                          [0, 0, 0], [7, 8, 9], [0, 0, 0]],
                         dtype=np.float32)
        def func(x):
            x_ = tf.reverse_sequence(x, seq_axis=0, batch_axis=1, seq_lengths=[9, 9, 9])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})
        x_val_shape = [5, 5, 7, 8, 9]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = tf.reverse_sequence(x, seq_axis=0, batch_axis=1, seq_lengths=[5, 5, 5, 5, 5])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite interpreter crashes on empty axis")
    @check_opset_min_version(10, "ReverseSequence")
    def test_reversev2_constant_axis(self):
        # Tests for constant axis.
        x_val_shape = [1, 2, 3, 4]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = reverse_v2(x, axis=[3])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        # Empty axis vector.
        x_val_shape = [2, 3, 4]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = reverse_v2(x, axis=[])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite reverse_v2 does not support multiple axes")
    @check_opset_min_version(10, "ReverseSequence")
    def test_reversev2_vector_axis(self):
        x_val_shape = [1, 2, 3, 4]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = reverse_v2(x, axis=[0, -3, 2, 3])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val_shape = [2, 3, 4]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = reverse_v2(x, axis=[-3, 1, 2])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val_shape = [5, 5, 9, 7, 8, 9]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = reverse_v2(x, axis=[0, 1, -2, 3, 5])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("tflite interpreter crashes on empty axis")
    @check_opset_min_version(10, "ReverseSequence")
    def test_reversev2_1D_tensor(self):
        # For tensors with 1 dimension and no axis to reverse.
        # Adds an identity block.
        x_val_shape = [4]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        def func(x):
            x_ = reverse_v2(x, axis=[0])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        def func(x):
            x_ = reverse_v2(x, axis=[])
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "GreaterEqual")
    def test_where(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.float32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.float32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.float32)
        def func(x):
            picks = tf.where(x > -1, true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array(1, dtype=np.float32)
        true_result = np.array(100, dtype=np.float32)
        false_result = np.array(-111, dtype=np.float32)
        def func(x):
            picks = tf.where(x > -1, true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "IsNaN")
    def test_where_isnan(self):
        x_val = np.array([1, 2, -3, float('nan'), -5, -6, float('nan'), 8, 9, 0], dtype=np.float32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.float32)
        def func(x):
            picks = tf.where(is_nan(x), true_result, x)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "IsNaN")
    def test_where_ismulinf(self):
        x_val1 = np.array([np.inf], dtype=np.float32)
        x_val2 = np.array([0], dtype=np.float32)
        true_result = np.array([np.inf], dtype=np.float32)
        def func(x1, x2):
            mul = tf.multiply(x1, x2)
            picks = tf.where(x1 < mul, true_result, x2)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_opset_min_version(9, "Where for strings needs opset 9")
    @skip_tfjs("Technically tf where doesn't support strings and tfjs doesn't like it")
    def test_where_string(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.float32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=str)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=str)
        def func(x):
            picks = tf.where(x > -1, true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "GreaterEqual")
    def test_where_bool(self):
        x_val = np.array([1, 2, -3, 4, -5], dtype=np.float32)
        true_result = np.array([True, False, True, False, True],
                               dtype=bool)
        false_result = np.array([False, True, False, True, True],
                                dtype=bool)
        def func(x):
            picks = tf.where(x > -1, true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "GreaterEqual")
    #@check_target("rs6", "onnxruntime Where type limitation")
    def test_where_int32(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.int32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.int32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.int32)
        def func(x):
            picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "GreaterEqual")
    @check_tf_max_version("1.15", "issues in tf-2.0, fix later")
    def test_where_with_two_rank_input(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.float32)
        true_result = np.array([[111, 111], [222, 222], [333, 333], [444, 444], [555, 555],
                                [666, 666], [777, 777], [888, 888], [999, 999], [1000, 1000]],
                               dtype=np.float32)
        false_result = np.array([[-111, -111], [-222, -222], [-333, -333], [-444, -444], [-555, -555],
                                 [-666, -666], [-777, -777], [-888, -888], [-999, -999], [-1000, -1000]],
                                dtype=np.float32)
        def func(x):
            cond = tf.greater_equal(x, 0)
            picks = tf.where(cond, true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "GreaterEqual")
    def test_where_with_two_rank_condition(self):
        x_val = np.array([[1, 2, -3, 4, -5, -6, -7, 8, 9, 0]], dtype=np.float32)
        true_result = np.array([[111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]],
                               dtype=np.float32)
        false_result = np.array([[-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000]],
                                dtype=np.float32)
        def func(x):
            picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "GreaterEqual")
    def test_where_with_three_rank_condition(self):
        x_val = np.array([[[1, 2, -3, 4, -5, -6, -7, 8, 9, 0]]], dtype=np.float32)
        true_result = np.array([[[111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]]],
                               dtype=np.float32)
        false_result = np.array([[[-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000]]],
                                dtype=np.float32)
        def func(x):
            picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "GreaterEqual")
    def test_where_scalar(self):
        x_val = np.array(6, dtype=np.float32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.float32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.float32)
        def func(x):
            picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
            return tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "NonZero")
    #@check_target("rs6", "onnxruntime Transpose type limitation")
    def test_where_with_cond_only(self):
        for np_type in [np.int32, np.float32]:
            x_val = np.random.randint(0, 2, size=[10, 20, 30]).astype(np_type)
            def func(x):
                # FIXME: was tf_placeholder(tf_type, shape=[None] * x_val.ndim, name=_TFINPUT)
                res = tf.where(x)
                return tf.identity(res, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14", "tf.strings.lower")
    @check_opset_min_version(10, "StringNormalizer")
    def test_string_lower(self):
        text_val1 = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=str)
        def func(text1):
            x = tf.strings.lower(text1)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1})

    @check_tf_min_version("1.14", "tf.strings.lower")
    @check_opset_min_version(10, "StringNormalizer")
    def test_string_lower_flat(self):
        text_val1 = np.array(["a", "Test 1 2 3", "♠♣", "Hi there", "test test", "♥♦"], dtype=str)
        def func(text1):
            x = tf.strings.lower(text1)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1})

    @check_tf_min_version("1.14", "tf.strings.lower")
    @check_opset_min_version(10, "StringNormalizer")
    def test_string_upper(self):
        text_val1 = np.array([["a", "Test 1 2 3", "♠♣"], ["Hi there", "test test", "♥♦"]], dtype=str)
        def func(text1):
            x = tf.strings.upper(text1)
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: text_val1})

    @check_opset_min_version(6, "cast")
    def test_shape_int32(self):
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.float32)
        def func(x):
            x_ = tf.multiply(x, x)
            x_ = tf.shape(x_, out_type=tf.int32)
            return tf.identity(x_, name=_TFOUTPUT)
        kwargs = {"check_dtype": True}
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, **kwargs)

    @unittest.skipIf(get_test_config().is_onnxruntime_backend and get_test_config().opset < 7,
                     "mul-1, mul-6 not supported in onnxruntime. conversion is covered since opset6")
    def test_shape_int64(self):
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.float32)
        def func(x):
            x_ = tf.multiply(x, x)
            x_ = tf.shape(x_, out_type=tf.int64)
            return tf.identity(x_, name=_TFOUTPUT)
        kwargs = {"check_dtype": True}
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, **kwargs)

    # @check_opset_min_version(7, "broadcasting op")
    @unittest.skip("disable it for now, since fold const has bug")
    def test_softmax_cross_entropy_with_logits(self):
        num_class = 5
        data_shape = [100, num_class]
        for np_dtype in [np.int32, np.int64]:
            label_val = np.random.randint(0, num_class - 1, data_shape).astype(np_dtype)
            logits_val = np.random.random(data_shape).astype(np.float32)

            def func(label, logits):
                res1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
                return tf.identity(res1, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val}, atol=1e-5)

    @check_opset_min_version(9, "sparse_softmax_cross_entropy_with_logits")
    def test_sparse_softmax_cross_entropy_with_logits(self):
        # FIXME: fails for opset 8 on onnxruntime-1.0, disable for now
        num_class = 5
        label_val = np.array([3, 2, 0, 4]).astype(np.int32)
        logits_val = np.random.random((len(label_val), num_class)).astype(np.float32)
        def func(label, logits):
            res1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
            return tf.identity(res1, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val})

    @check_target('rs6', 'SparseSoftmaxCrossEntropyWithLogits')
    def test_sparse_softmax_cross_entropy_with_logits_large_class(self):
        num_class = 30000
        label_val = np.array([3374, 2127, 10002, 48]).astype(np.int32)
        logits_val = np.random.random((len(label_val), num_class)).astype(np.float32)

        def func(label, logits):
            res = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
            return tf.identity(res, name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val}, rtol=1e-6)

    def test_matrix_band_part(self):
        input_val = np.random.randint(0, 666, (10, 15)).astype(np.int32)
        def func(input_x):
            res = tf.linalg.band_part(input_x, -1, 0)
            res1 = tf.linalg.band_part(input_x, 0, -1)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    def test_matrix_band_part_2(self):
        input_val = np.random.randint(0, 666, (1, 1)).astype(np.int32)
        def func(input_x):
            res = tf.linalg.band_part(input_x, -1, 0)
            res1 = tf.linalg.band_part(input_x, 0, -1)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    @check_opset_min_version(11, "CumSum")
    def test_matrix_band_part_3(self):
        for low, high in [(-1, 3), (2, 3), (4, 3), (0, -1), (0, 0), (-1, -1)]:
            input_val = np.random.randint(0, 666, (10, 15)).astype(np.int32)
            def func(input_x):
                res = tf.linalg.band_part(input_x, low, high)
                return tf.identity(res, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

    @check_opset_min_version(11, "CumSum")
    def test_matrix_band_part_4(self):
        for low, high in [(-1, 3), (2, 3), (4, 3), (0, -1), (0, 0)]:
            input_val = np.random.randint(0, 666, (2, 3, 10, 15)).astype(np.int32)
            def func(input_x):
                res = tf.linalg.band_part(input_x, low, high)
                return tf.identity(res, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

    @check_opset_min_version(11, "CumSum")
    def test_matrix_band_part_5(self):
        for low_val, high_val in [(2, 3), (4, 3), (0, 0), (2, 0)]:
            low_val = np.array(low_val, np.int32)
            high_val = np.array(high_val, np.int32)
            input_val = np.random.randint(0, 666, (2, 3, 10, 15)).astype(np.int32)
            def func(input_x, low, high):
                res = tf.linalg.band_part(input_x, low, high)
                return tf.identity(res, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: input_val, _INPUT1: low_val, _INPUT2: high_val})

    def test_matrix_band_part_bool(self):
        input_val = np.random.choice([False, True], size=(10, 15))
        def func(input_x):
            res = tf.linalg.band_part(input_x, -1, 0)
            res1 = tf.linalg.band_part(input_x, 0, -1)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    def test_floordiv(self):
        input_val_1 = np.random.random_sample(100).astype(np.int32)
        input_val_2 = (np.random.random_sample(100) + 1).astype(np.int32)
        def func(input_1, input_2):
            res = tf.math.floordiv(input_1, input_2)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

        input_val_1 = np.random.random_sample(100).astype(np.float32)
        input_val_2 = (np.random.random_sample(100) + 1).astype(np.float32)
        def func(input_1, input_2):
            res = tf.math.floordiv(input_1, input_2)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

        # test broadcasting
        input_val_1 = np.random.random_sample((10, 50)).astype(np.float32)
        input_val_2 = (np.random.random_sample(50) + 1).astype(np.float32)
        def func(input_1, input_2):
            res = tf.math.floordiv(input_1, input_2)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

    def test_floormod(self):
        input_val_1 = 100 * np.random.random_sample(100).astype(np.int32)
        input_val_2 = (100 * np.random.random_sample(100) + 1).astype(np.int32)
        def func(input_1, input_2):
            res = floormod(input_1, input_2)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

        input_val_1 = 100 * np.random.random_sample(100).astype(np.float32)
        input_val_2 = (100 * np.random.random_sample(100) + 1).astype(np.float32)
        def func(input_1, input_2):
            res = floormod(input_1, input_2)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2}, rtol=1e-5)

        # test broadcasting case
        input_val_1 = (50 * np.random.random_sample((10, 50)) + 1).astype(np.float32)
        input_val_2 = (50 * np.random.random_sample(50) + 1).astype(np.float32)
        def func(input_1, input_2):
            res = floormod(input_1, input_2)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2}, rtol=1e-4)

    def test_logical_not(self):
        input_val = np.random.randint(0, 2, (10, 20)).astype(bool)
        def func(x):
            res = tf.logical_not(x)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

    def test_reduce_all(self):
        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_all(input_tensor=x, keepdims=False)
            res1 = tf.reduce_all(input_tensor=x, axis=[0], keepdims=False)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(input_x):
            res = tf.reduce_all(input_tensor=input_x, keepdims=True)
            res1 = tf.reduce_all(input_tensor=input_x, axis=[0], keepdims=True)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    def test_reduce_any(self):
        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_any(input_tensor=x, keepdims=False)
            res1 = tf.reduce_any(input_tensor=x, axis=[0], keepdims=False)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_any(input_tensor=x, keepdims=True)
            res1 = tf.reduce_any(input_tensor=x, axis=[0], keepdims=True)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    @check_opset_min_version(11, "ReduceMin")
    def test_reduce_all_negative_axis(self):
        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_all(input_tensor=x, keepdims=False)
            res1 = tf.reduce_all(input_tensor=x, axis=[-1], keepdims=False)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(input_x):
            res = tf.reduce_all(input_tensor=input_x, keepdims=True)
            res1 = tf.reduce_all(input_tensor=input_x, axis=[-1], keepdims=True)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    @check_opset_min_version(11, "ReduceSum")
    def test_reduce_any_negative_axis(self):
        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_any(input_tensor=x, keepdims=False)
            res1 = tf.reduce_any(input_tensor=x, axis=[-1], keepdims=False)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_any(input_tensor=x, keepdims=True)
            res1 = tf.reduce_any(input_tensor=x, axis=[-1], keepdims=True)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    @check_opset_min_version(11, "ReduceSum")
    @check_tf_min_version("1.15")
    def test_reduce_any_empty_axis(self):
        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_any(input_tensor=x, keepdims=False)
            res1 = tf.reduce_any(input_tensor=x, axis=[], keepdims=False)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    def test_reduce_all_scalar_axis(self):
        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        def func(x):
            res = tf.reduce_all(input_tensor=x, keepdims=False)
            res1 = tf.reduce_all(input_tensor=x, axis=0, keepdims=False)
            return tf.identity(res, name=_TFOUTPUT), tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    @check_opset_min_version(13, "ReduceSum")
    @check_tf_min_version("1.15")
    def test_reduce_any_nonconst_axis(self):
        input_val = np.random.randint(0, 2, (2, 20)).astype(bool)
        y_val = np.array([1], np.int32)
        def func(x, y):
            res = tf.reduce_any(input_tensor=x, axis=y, keepdims=False)
            return tf.identity(res, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val, _INPUT1: y_val})

    @check_opset_min_version(7, "fill")
    def test_zeros_like(self):
        input_x = np.random.random_sample([10, 20]).astype(np.float32)
        input_y = np.array([20, 10]).astype(np.int64)

        def func(x, y):
            z = tf.reshape(x, y)
            return tf.zeros_like(z, name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x, _INPUT1: input_y})
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x.astype(np.int32), _INPUT1: input_y})

    @check_opset_min_version(8, "BroadcastTo")
    def test_zeros_like_bool(self):
        input_x = np.random.random_sample([10, 20]).astype(np.float32)
        input_y = np.array([20, 10]).astype(np.int64)

        def func(x, y):
            z = tf.reshape(x, y)
            return tf.zeros_like(z, name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x > 0.5, _INPUT1: input_y})

    @check_opset_min_version(9, "ConstantOfShape")
    def test_zeros_like_opset9(self):
        input_x = np.random.random_sample([3, 16, 16]).astype(np.float32)
        input_y = np.array([16, 16, 3]).astype(np.int64)

        def func(x, y):
            z = tf.reshape(x, y)
            return tf.zeros_like(z, name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x, _INPUT1: input_y})
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x.astype(np.int32), _INPUT1: input_y}, as_session=True,
                            graph_validator=lambda g: check_op_count(g, "ConstantOfShape", 1, disabled=False))

    def test_ones_like(self):
        input_x = np.random.random_sample([3, 16, 16]).astype(np.float32)
        input_y = np.array([16, 16, 3]).astype(np.int64)

        def func(x, y):
            z = tf.reshape(x, y)
            return tf.ones_like(z, name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x, _INPUT1: input_y})

    @check_opset_min_version(9, "is_nan")
    def test_isnan(self):
        # only compatible with dtype `float32`
        x_val1 = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32).reshape((2, 2))
        x_val3 = np.array([1.0, np.nan, -3.0, np.nan], dtype=np.float32).reshape((2, 2))
        for x_val in [x_val1, x_val2, x_val3]:
            def func(x):
                x_ = is_nan(x)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_ceil(self):
        x_val = np.array([-1.5, 1.2], dtype=np.float32)
        def func(x):
            x_ = tf.math.ceil(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_softplus(self):
        x_val = np.array([-1, 0, 1], dtype=np.float32)
        def func(x):
            x_ = tf.math.softplus(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_softsign(self):
        x_val = np.array([-1, 0, 1], dtype=np.float32)
        def func(x):
            x_ = tf.math.softsign(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_batch_to_spacend(self):
        block_size = [2, 2]
        crop = [[1, 0], [2, 1]]

        input_val = np.random.random_sample([40, 3, 5, 100]).astype(np.float32)
        def func(x):
            return batch_to_space_nd(x, block_size, crop, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

    @check_opset_min_version(11, "BatchToSpaceND")
    @unittest.skip("this was recently removed - but don't we want this to work ?")
    def test_batch_to_spacend_non_const(self):
        def func(input_x, block_shape, crops):
            return batch_to_space_nd(input_x, block_shape, crops, name=_TFOUTPUT)
        input_x_val = np.random.random_sample([40, 3, 5, 100]).astype(np.float32)  # NHWC
        block_shape_val = np.array([2, 2]).astype(np.int64)
        crops_val = np.array([[1, 0], [2, 1]]).astype(np.int64)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x_val, _INPUT1: block_shape_val, _INPUT2: crops_val})

    @check_opset_min_version(11, "SpaceToBatchND")
    @unittest.skip("this was recently removed - but don't we want this to work ?")
    def test_space_to_batchnd_non_const(self):
        input_x_val = np.random.random_sample([40, 5, 7, 66]).astype(np.float32)  # NHWC
        def func(input_x, block_size, pad):
            return batch_to_space_nd(input_x, block_size, pad, name=_TFOUTPUT)
        block_size_val = np.array([2, 2]).astype(np.int64)
        pad_val = np.array([[0, 1], [2, 1]]).astype(np.int64)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x_val, _INPUT1: block_size_val, _INPUT2: pad_val})

    @check_opset_min_version(11, "BatchToSpaceND")
    def test_batch_to_spacend_non_const_7d(self):
        x_type, y_type, z_type = np.float32, np.int64, np.int64
        # test 3D upto 7D input tensors
        for x_shape in [[12, 4, 4], [12, 4, 8, 3], [12, 4, 8, 3, 2], [12, 4, 8, 3, 2, 3], [12, 4, 8, 3, 2, 1, 3]]:
            # test 1D upto 2D block shapes
            for block_shape in [[2, 3], [2, 2], [2]]:
                # crop 1 layer at end of each dim
                # x and z can be dynamic.
                # y = block_shape cannot be dynamic without change to Transpose op spec
                crops = [[0, 1] for dim in block_shape]
                y_val = np.array(block_shape).astype(y_type)
                x_val = np.array([x + 1 for x in range(0, np.prod(x_shape))], dtype=x_type).reshape(x_shape)
                z_val = np.array(crops).astype(z_type)
                def func(x, z):
                    y = tf.constant(dtype=y_type, value=y_val, shape=y_val.shape, name=_TFINPUT1)
                    return batch_to_space_nd(x, y, z, name=_TFOUTPUT)
                self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT2: z_val})

    def test_depthwise_dilations_pattern(self):
        x_val = np.random.random_sample([1, 33, 34, 960]).astype(np.float32)
        kernel = np.random.random_sample([3, 3, 960, 1]).astype(np.float32)
        block_size = np.array([3, 3], np.int64)
        pad = np.array([[2, 4], [5, 3]])
        crop = np.array([[0, 0], [0, 0]])
        def func(x):
            y = space_to_batch_nd(x, block_size, pad)
            z = tf.nn.depthwise_conv2d(y, kernel, strides=[1, 1, 1, 1], padding='VALID')
            return batch_to_space_nd(z, block_size, crop, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "SpaceToBatchND")
    def test_space_to_batchnd_non_const_7d(self):
        x_type, y_type, z_type = np.float32, np.int64, np.int64
        # test 3D upto 7D input tensors
        for x_shape in [[2, 4, 4], [1, 4, 8, 3], [1, 4, 8, 3, 2], [1, 4, 8, 3, 2, 3], [1, 4, 8, 3, 2, 1, 3]]:
            # test 1D upto 2D block shapes
            for block_shape in [[2], [2, 2]]:
                # pad 1 layer at begin and end of each dim
                pads = [[1, 1] for dim in block_shape]
                y_val = np.array(block_shape).astype(y_type)
                x_val = np.array([x + 1 for x in range(0, np.prod(x_shape))], dtype=x_type).reshape(x_shape)
                z_val = np.array(pads).astype(z_type)
                # x and z can be dynamic.
                # y = block_shape cannot be dynamic without change to Transpose op spec
                def func(x, z):
                    y = tf.constant(dtype=y_type, value=y_val, shape=y_val.shape, name=_TFINPUT1)
                    return space_to_batch_nd(x, y, z, name=_TFOUTPUT)
                self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT2: z_val})

    @check_opset_min_version(10, "CropAndResize")
    def test_crop_and_resize(self):
        boxes_val = [[0.5, 0.7, 0.7, 0.9], [0.2, 0.4, 0.4, 0.6]]
        def func(input_x, box_ind):
            boxes = tf.constant(boxes_val, dtype=tf.float32)
            corp_size = tf.constant(np.array([20, 20]).astype(np.int32))
            return tf.image.crop_and_resize(input_x, boxes, box_ind, corp_size, name=_TFOUTPUT, method='bilinear')

        input_x_val = np.random.randint(low=0, high=256, size=[2, 36, 36, 3]).astype(np.float32)  # NHWC
        box_ind_val = np.array([1, 0]).astype(np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_x_val, _INPUT2: box_ind_val},
                            rtol=1e-04, atol=1e-03)

    @check_opset_min_version(11, "CropAndResize")
    def test_crop_and_resize_linear(self):
        def func(input_x, boxes, box_ind, corp_size):
            return tf.image.crop_and_resize(input_x, boxes, box_ind, corp_size, name=_TFOUTPUT, method='bilinear')

        input_x_val = np.random.randint(low=0, high=256, size=[2, 36, 36, 3]).astype(np.float32)  # NHWC
        boxes_val = np.array([[0.5, 0.7, 0.7, 0.9], [0.2, 0.4, 0.4, 0.6]]).astype(np.float32)
        box_ind_val = np.array([1, 0]).astype(np.int32)
        corp_size_val = np.array([20, 20]).astype(np.int32)
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: input_x_val, _INPUT1: boxes_val, _INPUT2: box_ind_val, _INPUT3: corp_size_val},
                            rtol=1e-05, atol=1e-04)

    @check_tf_min_version("1.9")
    @check_opset_min_version(11, "CropAndResize")
    def test_crop_and_resize_nearest(self):
        def func(input_x, boxes, box_ind, corp_size):
            return tf.image.crop_and_resize(input_x, boxes, box_ind, corp_size, name=_TFOUTPUT, method='nearest')
        input_x_val = np.random.randint(low=0, high=256, size=[1, 36, 36, 3]).astype(np.float32)  # NHWC
        boxes_val = np.array([[0.2, 0.4, 0.6, 0.8]]).astype(np.float32)
        box_ind_val = np.array([0]).astype(np.int32)
        corp_size_val = np.array([30, 30]).astype(np.int32)
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: input_x_val, _INPUT1: boxes_val, _INPUT2: box_ind_val, _INPUT3: corp_size_val},
                            rtol=1e-05, atol=1e-04)

    @check_opset_min_version(11, "CropAndResize")
    def test_crop_and_resize_extrapolation(self):
        def func(input_x, boxes, box_ind, corp_size):
            return tf.image.crop_and_resize(input_x, boxes, box_ind, corp_size, name=_TFOUTPUT, extrapolation_value=1.0)
        input_x_val = np.random.randint(low=0, high=256, size=[1, 36, 36, 3]).astype(np.float32)  # NHWC
        boxes_val = np.array([[0.2, 0.4, 1.2, 1.4]]).astype(np.float32)
        box_ind_val = np.array([0]).astype(np.int32)
        corp_size_val = np.array([40, 40]).astype(np.int32)
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: input_x_val, _INPUT1: boxes_val, _INPUT2: box_ind_val, _INPUT3: corp_size_val},
                            rtol=1e-04, atol=1e-03)

    @check_opset_min_version(11, "CropAndResize")
    def test_crop_and_resize_empty_tensor(self):
        def func(input_x, boxes, box_ind, corp_size):
            return tf.image.crop_and_resize(input_x, boxes, box_ind, corp_size, name=_TFOUTPUT, extrapolation_value=1.0)
        input_x_val = np.random.randint(low=0, high=256, size=[0, 36, 36, 3]).astype(np.float32)  # NHWC
        boxes_val = np.array([]).astype(np.float32).reshape([0, 4])
        box_ind_val = np.array([]).astype(np.int32)
        corp_size_val = np.array([40, 40]).astype(np.int32)
        self._run_test_case(func, [_OUTPUT],
                            {_INPUT: input_x_val, _INPUT1: boxes_val, _INPUT2: box_ind_val, _INPUT3: corp_size_val},
                            rtol=1e-04, atol=1e-03)

    def test_batch_to_space3d(self):
        block_size = [2, 2]
        crop = [[0, 1], [2, 1]]
        input_val = np.random.random_sample([40, 3, 100]).astype(np.float32)
        def func(x):
            return batch_to_space_nd(x, block_size, crop, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

    def test_space_to_batchnd(self):
        block_size = [2, 2]
        pad = [[0, 1], [2, 1]]
        input_val = np.random.random_sample([40, 5, 7, 66]).astype(np.float32)
        def func(x):
            return space_to_batch_nd(x, block_size, pad, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

        pad = [[0, 0], [1, 2]]
        input_val = np.random.random_sample([10, 6, 7, 66]).astype(np.float32)
        def func(x):
            return space_to_batch_nd(x, block_size, pad, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

    @check_opset_min_version(10, "is_inf")
    def test_isinf(self):
        x_types = [np.float32, np.float64]
        for x_type in x_types:
            x_val1 = np.array([1.0, -2.0, 3.0, -4.0], dtype=x_type)
            x_val2 = np.array([np.inf, np.inf, np.inf, np.inf], dtype=x_type).reshape((2, 2))
            x_val3 = np.array([1.0, np.inf, -3.0, np.inf, 5.0, np.inf, -7.0, np.inf], dtype=x_type).reshape((2, 2, 2))
            for x_val in [x_val1, x_val2, x_val3]:
                def func(x):
                    x_ = is_inf(x)
                    return tf.identity(x_, name=_TFOUTPUT)
                self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("2.3")
    @check_opset_min_version(10, "NonMaxSuppression")
    def test_non_max_suppression_v2(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func(boxes, scores):
            res1 = tf.raw_ops.NonMaxSuppressionV2(boxes=boxes, scores=scores,
                                                  max_output_size=int(box_num / 2), iou_threshold=0.5)
            res2 = tf.raw_ops.NonMaxSuppressionV2(boxes=boxes, scores=scores,
                                                  max_output_size=0, iou_threshold=0.5)
            return tf.identity(res1, name=_TFOUTPUT), tf.identity(res2, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_tf_min_version("2.3")
    @check_opset_min_version(10, "NonMaxSuppression")
    def test_non_max_suppression_v3(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func(boxes, scores):
            res1 = tf.raw_ops.NonMaxSuppressionV3(boxes=boxes, scores=scores, score_threshold=0.1,
                                                  max_output_size=int(box_num / 2), iou_threshold=0.5)
            res2 = tf.raw_ops.NonMaxSuppressionV3(boxes=boxes, scores=scores, score_threshold=0.1,
                                                  max_output_size=0, iou_threshold=0.5)
            return tf.identity(res1, name=_TFOUTPUT), tf.identity(res2, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_tf_min_version("2.3")
    @check_opset_min_version(10, "NonMaxSuppression")
    @skip_tfjs("TFJS executes model incorrectly")
    def test_non_max_suppression_v4(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func1(boxes, scores):
            res1, res2 = tf.raw_ops.NonMaxSuppressionV4(boxes=boxes, scores=scores, score_threshold=0.1,
                                                        max_output_size=int(box_num / 2), iou_threshold=0.5)
            return tf.identity(res1, name=_TFOUTPUT), tf.identity(res2, name=_TFOUTPUT1)

        self._run_test_case(func1, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

        def func2(boxes, scores):
            res1, res2 = tf.raw_ops.NonMaxSuppressionV4(boxes=boxes, scores=scores, score_threshold=0.1,
                                                        max_output_size=2 * box_num, iou_threshold=0.5,
                                                        pad_to_max_output_size=True)
            return tf.identity(res1, name=_TFOUTPUT), tf.identity(res2, name=_TFOUTPUT1)

        self._run_test_case(func2, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_tf_min_version("2.3")
    @check_opset_min_version(10, "NonMaxSuppression")
    def test_non_max_suppression_v5(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func1(boxes, scores):
            res1, res2, res3 = tf.raw_ops.NonMaxSuppressionV5(boxes=boxes, scores=scores, score_threshold=0.1,
                                                              max_output_size=int(box_num / 2), iou_threshold=0.5,
                                                              soft_nms_sigma=0)
            return tf.identity(res1, name=_TFOUTPUT), tf.identity(res2, name=_TFOUTPUT1), \
                   tf.identity(res3, name=_TFOUTPUT2)

        self._run_test_case(func1, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: boxes_val, _INPUT1: scores_val})

        def func2(boxes, scores):
            res1, res2, res3 = tf.raw_ops.NonMaxSuppressionV5(boxes=boxes, scores=scores, score_threshold=0.1,
                                                              max_output_size=2 * box_num, iou_threshold=0.5,
                                                              soft_nms_sigma=0, pad_to_max_output_size=True)
            return tf.identity(res1, name=_TFOUTPUT), tf.identity(res2, name=_TFOUTPUT1), \
                   tf.identity(res3, name=_TFOUTPUT2)

        self._run_test_case(func2, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_opset_min_version(10, "NonMaxSuppression")
    def test_non_max_suppression(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func(boxes, scores):
            res1 = tf.image.non_max_suppression(boxes, scores, max_output_size=int(box_num / 2))
            res2 = tf.image.non_max_suppression(boxes, scores, max_output_size=0)
            return tf.identity(res1, name=_TFOUTPUT), tf.identity(res2, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_opset_min_version(10, "NonMaxSuppression")
    @allow_missing_shapes("TF shape inference misses reshape to scalar")
    @skip_tfjs("TFJS executes model incorrectly")
    def test_non_max_suppression_v4_padded(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func(boxes, scores):
            ret1, ret2 = tf.image.non_max_suppression_padded(boxes, scores, max_output_size=int(box_num * 2),
                                                             pad_to_max_output_size=True)
            return tf.identity(ret1, name=_TFOUTPUT), tf.identity(ret2, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_opset_min_version(10, "NonMaxSuppression")
    @allow_missing_shapes("TF shape inference misses reshape to scalar")
    @skip_tfjs("TFJS executes model incorrectly")
    def test_non_max_suppression_v4_no_padding(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func(boxes, scores):
            ret1, ret2 = tf.image.non_max_suppression_padded(boxes, scores, max_output_size=int(box_num),
                                                             pad_to_max_output_size=False)
            return tf.identity(ret1, name=_TFOUTPUT), tf.identity(ret2, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_tf_min_version("1.15")
    @check_opset_min_version(10, "NonMaxSuppression")
    def test_non_max_suppression_v5(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)

        def func(boxes, scores):
            ret1, ret2 = tf.image.non_max_suppression_with_scores(boxes, scores, max_output_size=int(box_num / 2),
                                                                  soft_nms_sigma=0.0)
            return tf.identity(ret1, name=_TFOUTPUT), tf.identity(ret2, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_tf_min_version("2.3")
    @check_opset_min_version(12, "GatherND with batch_dims")
    def test_combined_non_max_suppression_pad_and_clip(self):
        batch_size = 8
        box_num = 10
        classes_num = 2
        max_total_size = 9
        boxes_val = np.random.random_sample([batch_size, box_num, 1, 4]).astype(np.float32) * 2 - 0.5
        scores_val = np.random.random_sample([batch_size, box_num, classes_num]).astype(np.float32)

        def func(boxes, scores):
            nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = \
                tf.image.combined_non_max_suppression(boxes=boxes, scores=scores, score_threshold=0.1,
                                                      max_output_size_per_class=3, max_total_size=max_total_size,
                                                      iou_threshold=0.5, pad_per_class=True, clip_boxes=True)
            out1 = tf.identity(nmsed_boxes, name=_TFOUTPUT)
            out2 = tf.identity(nmsed_scores, name=_TFOUTPUT1)
            out3 = tf.identity(nmsed_classes, name=_TFOUTPUT2)
            out4 = tf.identity(valid_detections, name=_TFOUTPUT3)
            return out1, out2, out3, out4

        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2, _OUTPUT3], {_INPUT: boxes_val, _INPUT1: scores_val})

    @check_tf_min_version("2.3")
    @check_opset_min_version(12, "GatherND with batch_dims")
    def test_combined_non_max_suppression_no_pad_no_clip(self):
        batch_size = 8
        box_num = 10
        classes_num = 2
        max_total_size = 9
        boxes_val = np.random.random_sample([batch_size, box_num, 1, 4]).astype(np.float32) * 2 - 0.5
        scores_val = np.random.random_sample([batch_size, box_num, classes_num]).astype(np.float32)

        def func(boxes, scores):
            nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = \
                tf.image.combined_non_max_suppression(boxes=boxes, scores=scores, score_threshold=0.1,
                                                      max_output_size_per_class=3, max_total_size=max_total_size,
                                                      iou_threshold=0.5, pad_per_class=False, clip_boxes=False)
            out1 = tf.identity(nmsed_boxes, name=_TFOUTPUT)
            out2 = tf.identity(nmsed_scores, name=_TFOUTPUT1)
            out3 = tf.identity(nmsed_classes, name=_TFOUTPUT2)
            out4 = tf.identity(valid_detections, name=_TFOUTPUT3)
            return out1, out2, out3, out4

        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2, _OUTPUT3], {_INPUT: boxes_val, _INPUT1: scores_val})

    def _conv1d_test(self, x_val, w, stride=None, padding="VALID", rtol=1e-07):
        if stride is None:
            stride = 1
        def func(x):
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv1d(x, kernel, stride=stride, padding=padding)
            return tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=rtol)

    def test_conv1d_1(self):
        x_val = make_xval((1, 7, 1))
        w = np.array([2., 1., 3.], dtype=np.float32).reshape(3, 1, 1)
        self._conv1d_test(x_val, w)

    def test_conv1d_2(self):
        x_val = make_xval((1, 7, 1))
        w = np.array([2., 1., 3.], dtype=np.float32).reshape(3, 1, 1)
        self._conv1d_test(x_val, w, stride=2)

    def test_conv1d_3(self):
        x_val = make_xval((1, 7, 1))
        w = np.array([2., 1., 3.], dtype=np.float32).reshape(3, 1, 1)
        self._conv1d_test(x_val, w, padding="SAME")

    def test_conv1d_4(self):
        x_val = make_xval((1, 7, 1))
        w = np.array([2., 1., 3.], dtype=np.float32).reshape(3, 1, 1)
        self._conv1d_test(x_val, w, rtol=1e-05)

    def test_conv1d_5(self):
        x_val = make_xval((1, 7, 1))
        w = np.array([3., 3., 3.], dtype=np.float32).reshape(3, 1, 1)
        self._conv1d_test(x_val, w)

    @check_opset_min_version(10, "ThresholdedRelu")
    def test_thresholded_relu(self):
        # tf.keras.layers.ThresholdedReLU only supports `float32` for x
        x_val = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32).reshape((3, 3))
        theta_vals = [0.0, 0.5, 1.0, 2.0]
        for theta_val in theta_vals:
            def func(x):
                t = tf.keras.layers.ThresholdedReLU(theta=theta_val)
                x_ = t.call(x)
                return tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case(func, [_OUTPUT], {_INPUT: x_val},
                                graph_validator=lambda g: check_op_count(g, "ThresholdedRelu", 1))

    @check_tf_min_version("1.13")
    @check_opset_min_version(11, "MaxPoolWithArgmax")
    def test_maxpoolwithargmax(self):
        for p in get_maxpoolwithargmax_getdata():
            _, padding, x_shape, ksize, strides = p
            x_val = np.random.uniform(0, 10, x_shape)
            def func(x):
                mp = tf.nn.max_pool_with_argmax(x, ksize, strides, padding=padding)
                return tf.identity(mp[0], name=_TFOUTPUT), tf.identity(mp[1], name=_TFOUTPUT1)
            self.logger.debug(str(p))
            self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val})

    @check_tf_min_version("1.15")
    @check_opset_min_version(11, "MaxPoolWithArgmax")
    def test_maxpoolwithargmax_batch_in_index(self):
        padding = 'SAME'
        x_shape = [2, 10, 5, 3]
        ksize = [1, 4, 4, 1]
        strides = [1, 1, 1, 1]
        x_val = np.random.uniform(0, 10, x_shape)
        def func(x):
            mp = tf.nn.max_pool_with_argmax(x, ksize, strides, padding=padding, include_batch_in_index=True)
            return tf.identity(mp[0], name=_TFOUTPUT), tf.identity(mp[1], name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val})

    @check_tf_min_version("1.15")
    @check_opset_min_version(11, "MaxPoolWithArgmax")
    def test_maxpoolwithargmax_unknown_c(self):
        padding = 'SAME'
        x_shape = [2, 10, 5, 1]
        ksize = [1, 4, 4, 1]
        strides = [1, 1, 1, 1]
        x_val = np.random.uniform(0, 10, x_shape)
        s_val = np.array([2, 10, 5, 4], np.int64)
        def func(x, s):
            x = tf.broadcast_to(x, s)
            mp = tf.nn.max_pool_with_argmax(x, ksize, strides, padding=padding, include_batch_in_index=True)
            return tf.identity(mp[0], name=_TFOUTPUT), tf.identity(mp[1], name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val, _INPUT1: s_val})

    @check_opset_min_version(10, "Selu")
    def test_selu(self):
        x_val = np.random.random_sample([3]).astype(np.float32)
        def func(x):
            y = tf.nn.selu(x)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "ClipByValue (needs broadcast)")
    def test_clip_by_value(self):
        # float32, dynamic min/max
        x_val = np.arange(0, 24, dtype=np.float32).reshape([3, 8])
        x_minval = np.array(8.5, dtype=np.float32)
        x_maxval = np.array(16.5, dtype=np.float32)
        def func(x, x_min, x_max):
            y = tf.clip_by_value(x, x_min, x_max)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: x_minval, _INPUT2: x_maxval})

        # float32, const min/max
        x_val = np.arange(0, 24, dtype=np.float32).reshape([3, 8])
        def func(x):
            y = tf.clip_by_value(x, 8.5, 16.5)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        # int32, converter needs to cast, const min/max
        x_val = np.arange(0, 24, dtype=np.int32).reshape([3, 8])
        def func(x):
            y = tf.clip_by_value(x, 8, 16)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_softmax(self):
        x_val = np.arange(0, 24, dtype=np.float32).reshape([3, 1, 8])
        def func(x):
            y = tf.nn.softmax(x)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_log_softmax(self):
        x_val = np.arange(0, 24, dtype=np.float32).reshape([3, 1, 8])
        def func(x):
            y = tf.nn.log_softmax(x)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Range")
    def test_ctc_greedy_decoder(self):
        x_val = np.random.uniform(size=(3, 4, 5)).astype(np.float32)
        s_val = np.array([3, 3, 2, 3], np.int32)
        def func(x, s):
            [decoded], logits = tf.nn.ctc_greedy_decoder(x, s, merge_repeated=False)
            r1 = tf.identity(decoded.indices, name=_TFOUTPUT)
            r2 = tf.identity(decoded.values, name=_TFOUTPUT1)
            r3 = tf.identity(decoded.dense_shape, name=_TFOUTPUT2)
            r4 = tf.identity(logits, name=_TFOUTPUT3)
            return r1, r2, r3, r4
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2, _OUTPUT3], {_INPUT: x_val, _INPUT1: s_val})

    @check_opset_min_version(11, "Range")
    def test_ctc_greedy_decoder_merge_repeated(self):
        x_val = np.random.uniform(size=(6, 4, 5)).astype(np.float32)
        s_val = np.array([5, 6, 4, 6], np.int32)
        def func(x, s):
            [decoded], logits = tf.nn.ctc_greedy_decoder(x, s, merge_repeated=True)
            r1 = tf.identity(decoded.indices, name=_TFOUTPUT)
            r2 = tf.identity(decoded.values, name=_TFOUTPUT1)
            r3 = tf.identity(decoded.dense_shape, name=_TFOUTPUT2)
            r4 = tf.identity(logits, name=_TFOUTPUT3)
            return r1, r2, r3, r4
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2, _OUTPUT3], {_INPUT: x_val, _INPUT1: s_val})

    # test for gemm pattern0: alpha*A*B + beta*C
    def test_gemm_pattern0(self):
        max_number = 10
        m = np.random.randint(max_number)
        n = np.random.randint(max_number)
        k = np.random.randint(max_number)
        x_val1 = np.random.rand(m, n).astype("float32")
        x_val2 = np.random.rand(n, k).astype("float32")
        x_val3 = np.random.rand(m, k).astype("float32")
        def func(a, b, c):
            alpha = tf.constant(1.0, dtype=tf.float32)
            beta = tf.constant(2.0, dtype=tf.float32)
            mul1 = tf.multiply(alpha, tf.matmul(a, b))
            mul2 = tf.multiply(beta, c)
            x_ = mul1 + mul2
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, _INPUT2: x_val3},
                            graph_validator=lambda g: check_op_count(g, "Gemm", 1))

    # test for gemm pattern1: alpha*A*B + C
    def test_gemm_pattern1(self):
        max_number = 10
        m = np.random.randint(max_number)
        n = np.random.randint(max_number)
        k = np.random.randint(max_number)
        x_val1 = np.random.rand(m, n).astype("float32")
        x_val2 = np.random.rand(n, k).astype("float32")
        x_val3 = np.random.rand(m, k).astype("float32")
        def func(a, b, c):
            alpha = tf.constant(1.0, dtype=tf.float32)
            x_ = tf.multiply(alpha, tf.matmul(a, b)) + c
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, _INPUT2: x_val3},
                            graph_validator=lambda g: check_op_count(g, "Gemm", 1))

    # test for gemm pattern2: A*B + beta*C
    def test_gemm_pattern2(self):
        max_number = 10
        m = np.random.randint(max_number)
        n = np.random.randint(max_number)
        k = np.random.randint(max_number)
        x_val1 = np.random.rand(m, n).astype("float32")
        x_val2 = np.random.rand(n, k).astype("float32")
        x_val3 = np.random.rand(m, k).astype("float32")
        def func(a, b, c):
            beta = tf.constant(2.0, dtype=tf.float32)
            x_ = tf.matmul(a, b) + tf.multiply(beta, c)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, _INPUT2: x_val3},
                            graph_validator=lambda g: check_op_count(g, "Gemm", 1))

    # test for gemm pattern3: A*B + C
    def test_gemm_pattern3(self):
        max_number = 10
        m = np.random.randint(max_number)
        n = np.random.randint(max_number)
        k = np.random.randint(max_number)
        x_val1 = np.random.rand(m, n).astype("float32")
        x_val2 = np.random.rand(n, k).astype("float32")
        x_val3 = np.random.rand(m, k).astype("float32")
        def func(a, b, c):
            x_ = tf.matmul(a, b) + c
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, _INPUT2: x_val3},
                            graph_validator=lambda g: check_op_count(g, "Gemm", 1))

    # test for gemm pattern4: A*B + C [addbias] - 1D bias!
    def test_gemm_pattern4(self):
        max_number = 10
        m = np.random.randint(max_number)
        n = np.random.randint(max_number)
        k = np.random.randint(max_number) # bias add requires 1D tensor
        x_val1 = np.random.rand(m, n).astype("float32")
        x_val2 = np.random.rand(n, k).astype("float32")
        x_val3 = np.random.rand(k).astype("float32")
        def func(a, b, c):
            x_ = tf.nn.bias_add(tf.matmul(a, b), c)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, _INPUT2: x_val3},
                            graph_validator=lambda g: check_op_count(g, "Gemm", 1))

    # test for gemm pattern0: alpha*A*B + beta*C
    @check_opset_min_version(12, "Optimizer bug in ORT 1.2")
    def test_gemm_pattern0_fail_broadcast(self):
        # shapes (3, 3) * (3, 1) + (1, 4) => (3, 1) + (1, 4)
        # c not uni-broadcastable to a * b, so should not use GEMM
        m, n, k = 3, 3, 1
        x_val1 = np.random.rand(m, n).astype("float32")
        x_val2 = np.random.rand(n, k).astype("float32")
        x_val3 = np.random.rand(k, 4).astype("float32")

        def func(a, b, c):
            alpha = tf.constant(1.0, dtype=tf.float32)
            beta = tf.constant(2.0, dtype=tf.float32)
            mul1 = tf.multiply(alpha, tf.matmul(a, b))
            mul2 = tf.multiply(beta, c)
            x_ = mul1 + mul2
            return tf.identity(x_, name=_TFOUTPUT)

        def graph_validator(g):
            if 'Gemm' in [n.type for n in g.get_nodes()]: return False
            return True

        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, _INPUT2: x_val3},
                            graph_validator=graph_validator)

    def test_graph_matcher(self):
        shape = [2, 6]
        x_val = np.random.random(shape).astype(np.float32)
        y_val = np.random.random(shape).astype(np.float32)
        z_val = np.random.random(shape).astype(np.float32)
        def func(x, y, z):
            tmp1 = x + y
            tmp2 = x - y
            tmp3 = tf.multiply(tmp1, z)
            tmp4 = tf.multiply(tmp2, z)
            return tf.add(tmp4, tmp3, name=_TFOUTPUT)

        onnx_graph = self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})
        pattern = \
            OpTypePattern('Add', name='output', inputs=[
                OpTypePattern('Mul', inputs=[
                    OpTypePattern('Add', name='input1'),
                    OpTypePattern('*', name='input2')]),
                OpTypePattern('Mul', inputs=[
                    OpTypePattern('Sub', name='input1'),
                    OpTypePattern('*', name='input2')])])

        matcher = GraphMatcher(pattern, allow_reorder=False)
        match_results = list(matcher.match_ops(onnx_graph.get_nodes()))
        self.assertTrue(len(match_results) == 0)
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(onnx_graph.get_nodes()))
        self.assertTrue(len(match_results) == 1)

    def test_add2(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.add(x, x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "CumSum")
    def test_cumsum(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.cumsum(x, axis=1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "CumSum")
    def test_cumsum_axis1_reverse_exclusive(self):
        x_val = np.array([1., 2., 3., 4.,
                          5., 6., 7., 8.,
                          9., 10., 11., 12.,
                          13., 14., 15., 16.,
                          17., 18., 19., 20.,
                          21., 22., 23., 24.], dtype=np.float32).reshape((2, 3, 4))
        def func(x):
            x_ = tf.cumsum(x, axis=1, reverse=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_cumprod(self):
        x_val = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape((2, 2))
        def func(x):
            x_ = tf.math.cumprod(x, axis=0)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_cumprod_axis1(self):
        x_val = np.array([1., 2., 3., 4.,
                          5., 6., 7., 8.,
                          9., 10., 11., 12.,
                          13., 14., 15., 16.,
                          17., 18., 19., 20.,
                          21., 22., 23., 24.], dtype=np.float32).reshape((2, 3, 4))
        def func(x):
            x_ = tf.math.cumprod(x, axis=1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_cumprod_axis1_reverse(self):
        x_val = np.array([1., 2., 3., 4.,
                          5., 6., 7., 8.,
                          9., 10., 11., 12.,
                          13., 14., 15., 16.,
                          17., 18., 19., 20.,
                          21., 22., 23., 24.], dtype=np.float32).reshape((2, 3, 4))
        def func(x):
            x_ = tf.math.cumprod(x, axis=1, reverse=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_cumprod_axis1_reverse_exclusive(self):
        x_val = np.array([1., 2., 3., 4.,
                          5., 6., 7., 8.,
                          9., 10., 11., 12.,
                          13., 14., 15., 16.,
                          17., 18., 19., 20.,
                          21., 22., 23., 24.], dtype=np.float32).reshape((2, 3, 4))
        def func(x):
            x_ = tf.math.cumprod(x, axis=1, reverse=True, exclusive=True)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Round")
    def test_round(self):
        x_val = np.array([-0.7, -0.5, -0.0, 0.0, +0.0, 0.3, 0.5, 0.7, float('nan')], dtype=np.float32)
        def func(x):
            x_ = tf.round(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    def test_round_approx(self):
        # In lower opsets there is no Round, but we can approximate it forgoing nearest even
        x_val = np.array([-0.7, -0.5, -0.0, 0.0, +0.0, 0.3, 1.5, 0.7, float('nan')], dtype=np.float32)
        def func(x):
            x_ = tf.round(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Round")
    def test_rint(self):
        x_val = np.array([-2.7, -1.5, -0.0, +0.0, 0.3, 0.5, 1.5, 2.5, 3.4, 3.5, float('nan')], dtype=np.float32)
        def func(x):
            x_ = tf.math.rint(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Det")
    @unittest.skip("unclear how this is called in tf-2, fix later")
    def test_determinant(self):
        x_val = np.array([1., 2., 3., 4., 1., 2.,
                          2., 1., 1., 3., 3., 1.,
                          1., 2., 3., 4., 1., 2.,
                          2., 1., 1., 3., 3., 1.],
                         dtype=np.float32).reshape((1, 2, 3, 2, 2))
        def func(x):
            x_ = tf.matrix_determinant(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "BitShift")
    def test_bitshift_left(self):
        x_val = np.array([16, 4, 1], dtype=np.int32)
        y_val = np.array([1, 2, 3], dtype=np.int32)
        def func(x, y):
            x_ = tf.bitwise.left_shift(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(11, "BitShift")
    def test_bitshift_right(self):
        info = np.iinfo(np.int32)
        x_val = np.array([-1, 0, 1, info.max, info.min], dtype=np.int32)
        def func(x):
            x_ = tf.bitwise.right_shift(x, 1)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(18, "BitwiseAnd")
    def test_bitwise_and(self):
        x_val = np.array([21, 4, 1], dtype=np.int32)
        y_val = np.array([45, 69, 3], dtype=np.int32)
        def func(x, y):
            x_ = tf.bitwise.bitwise_and(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(18, "BitwiseOr")
    def test_bitwise_or(self):
        x_val = np.array([21, 4, 87], dtype=np.int32)
        y_val = np.array([45, 69, 173], dtype=np.int32)
        def func(x, y):
            x_ = tf.bitwise.bitwise_or(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(18, "BitwiseXor")
    def test_bitwise_xor(self):
        x_val = np.array([21, 4, 87], dtype=np.int32)
        y_val = np.array([45, 69, 173], dtype=np.int32)
        def func(x, y):
            x_ = tf.bitwise.bitwise_xor(x, y)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(18, "BitwiseNot")
    def test_bitwise_not(self):
        x_val = np.array([21, 4, 1], dtype=np.int32)
        def func(x):
            x_ = tf.bitwise.invert(x)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14", "tensor_scatter_nd_update needs tf 1.14")
    @check_opset_min_version(11, "ScatterND")
    def test_tensor_scatter_update(self):
        x_val = np.array([10, 20, 30, 40], dtype=np.int32).reshape((4))
        y_val = np.array([0, 2], dtype=np.int64).reshape((2, 1))
        z_val = np.array([8, 11], dtype=np.int32).reshape((2))

        def func(x, y, z):
            x_ = tf.tensor_scatter_nd_update(x, y, z)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_tf_min_version("1.15", "tensor_scatter_nd_update for strings needs tf 1.15")
    @check_opset_min_version(11, "ScatterND")
    @skip_tflite("Conversion crashes")
    def test_tensor_scatter_update_str(self):
        x_val = np.array(['A', '♠♣♥♦', 'B', 'C'], dtype=str).reshape((4))
        y_val = np.array([0, 2], dtype=np.int64).reshape((2, 1))
        z_val = np.array(['☺', '11'], dtype=str).reshape((2))

        def func(x, y, z):
            x_ = tf.tensor_scatter_nd_update(x, y, z)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_tf_min_version("1.15", "tensor_scatter_nd_update for strings needs tf 1.15")
    @check_opset_min_version(11, "ScatterND")
    @skip_tflite("Conversion crashes")
    def test_tensor_scatter_update_str_const(self):
        x_val = np.array(['A', '♠♣♥♦', 'B', 'C'], dtype=str).reshape((4))
        y_val = np.array([0, 2], dtype=np.int64).reshape((2, 1))
        z_val = np.array(['☺', '11'], dtype=str).reshape((2))

        def func(x, y):
            z = tf.constant(z_val)
            x_ = tf.tensor_scatter_nd_update(x, y, z)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_tf_min_version("1.14", "tensor_scatter_nd_update needs tf 1.14")
    @check_opset_min_version(11, "ScatterND")
    def test_tensor_scatter_update_cast_indices(self):
        x_val = np.array([10, 20, 30, 40], dtype=np.int32).reshape((4))
        y_val = np.array([0, 2], dtype=np.int32).reshape((2, 1))
        z_val = np.array([8, 11], dtype=np.int32).reshape((2))

        def func(x, y, z):
            x_ = tf.tensor_scatter_nd_update(x, y, z)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_opset_min_version(16, "ScatterND")
    def test_scatternd_add(self):
        x_val = np.array([10, 20, 30, 40], dtype=np.int32).reshape((4))
        y_val = np.array([0, 2], dtype=np.int64).reshape((2, 1))
        z_val = np.array([20, 30], dtype=np.int32).reshape((2))

        def func(x, y, z):
            x_ = tf.tensor_scatter_nd_add(x, y, z)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_opset_min_version(11, "ScatterND")
    def test_scatternd_1d(self):
        x_val = np.array([4, 3, 1, 7], dtype=np.int32).reshape((4, 1))
        y_val = np.array([9, 10, 11, 12], dtype=np.int64).reshape((4))
        z_val = np.array([8], dtype=np.int32).reshape(1)

        def func(x, y, z):
            x_ = tf.scatter_nd(x, y, z)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_opset_min_version(11, "ScatterND")
    def test_scatternd_3d(self):
        x_val = np.array([0, 2], dtype=np.int32).reshape((2, 1))
        y_val = np.array([[[5, 5, 5, 5], [6, 6, 6, 6],
                           [7, 7, 7, 7], [8, 8, 8, 8]],
                          [[5, 5, 5, 5], [6, 6, 6, 6],
                           [7, 7, 7, 7], [8, 8, 8, 8]]], dtype=np.float32).reshape((2, 4, 4))
        z_val = np.array([4, 4, 4], dtype=np.int32).reshape(3)

        def func(x, y, z):
            x_ = tf.scatter_nd(x, y, z)
            return tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val, _INPUT2: z_val})

    @check_opset_min_version(11, "Unique")
    def test_unique(self):
        x_val = np.array([1, 2, 8, 1, 2, 2, 7, 7, 7, 1], dtype=np.float32)
        def func(x):
            x1_, _ = tf.unique(x)
            y1 = tf.identity(x1_, name=_TFOUTPUT)
            return y1
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Unique")
    def test_unique_indices_int64(self):
        x_val = np.array([2, 3, 3, 6, 4, 1, 1], dtype=np.float32)
        def func(x):
            x1_, x2_ = tf.unique(x, out_idx=tf.int64)
            y1 = tf.identity(x1_, name=_TFOUTPUT)
            y2 = tf.identity(x2_, name=_TFOUTPUT1)
            return y1, y2
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val})

    @check_opset_min_version(11, "Unique")
    def test_unique_indices_int32(self):
        x_val = np.array([2, 3, 3, 6, 4, 1, 1], dtype=np.float32)
        def func(x):
            x1_, x2_ = tf.unique(x, out_idx=tf.int32)
            y1 = tf.identity(x1_, name=_TFOUTPUT)
            y2 = tf.identity(x2_, name=_TFOUTPUT1)
            return y1, y2
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val})

    @check_opset_min_version(11, "Unique")
    def test_unique_with_counts(self):
        x_val = np.array([1, 2, 8, 1, 2, 2, 7, 7, 7, 1], dtype=np.float32)
        def func(x):
            x1_, x2_, x3_ = tf.unique_with_counts(x)
            y1 = tf.identity(x1_, name=_TFOUTPUT)
            y2 = tf.identity(x2_, name=_TFOUTPUT1)
            y3 = tf.identity(x3_, name=_TFOUTPUT2)
            return y1, y2, y3
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: x_val})

    @check_opset_min_version(11, "Unique")
    def test_unique_with_counts_out_int64(self):
        x_val = np.array([2, 3, 3, 6, 4, 1, 1], dtype=np.float32)
        def func(x):
            x1_, x2_, x3_ = tf.unique_with_counts(x, out_idx=tf.int64)
            y1 = tf.identity(x1_, name=_TFOUTPUT)
            y2 = tf.identity(x2_, name=_TFOUTPUT1)
            y3 = tf.identity(x3_, name=_TFOUTPUT2)
            return y1, y2, y3
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: x_val})

    @check_opset_min_version(11, "Unique")
    def test_unique_with_counts_out_int32(self):
        x_val = np.array([2, 3, 3, 6, 4, 1, 1], dtype=np.float32)
        def func(x):
            x1_, x2_, x3_ = tf.unique_with_counts(x, out_idx=tf.int32)
            y1 = tf.identity(x1_, name=_TFOUTPUT)
            y2 = tf.identity(x2_, name=_TFOUTPUT1)
            y3 = tf.identity(x3_, name=_TFOUTPUT2)
            return y1, y2, y3
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: x_val})

    @check_opset_min_version(11, "Unique")
    def test_bincount(self):
        x_val = np.array([5, 2, 3, 1, 3, 2, 7, 5, 9, 10], dtype=np.int32)
        def func(x):
            x_ = tf.math.bincount(x)
            y_ = tf.identity(x_, name=_TFOUTPUT)
            return y_
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tflite("Bug in tflite output shapes")
    @skip_tfjs("TFJS executes model incorrectly")
    @check_opset_min_version(11, "Unique")
    @check_tf_min_version("2.3", "needs tf.math.bincount with axis attr")
    def test_dense_bincount(self):
        x_val = np.array([[5, 2, 3, 1, 3], [2, 7, 5, 9, 10]], dtype=np.int32)
        y_val = np.array([[2.0, 1.5, 3.5, 4.5, 5.5], [6.5, 7.5, 8.5, 9.5, 10.5]], dtype=np.float32)
        for a in [0, -1]:
            for b in [True, False]:
                def func(x, y):
                    x_ = tf.math.bincount(x, axis=a, binary_output=b)
                    y_ = tf.identity(x_, name=_TFOUTPUT)
                    return y_
                self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(11, "ScatterND")
    def test_sparse_to_dense(self):
        i_val = np.array([[0, 0, 0], [0, 0, 2], [0, 1, 3], [1, 2, 2], [1, 2, 3]], dtype=np.int64)
        v_val = np.array([1.5, 1.6, 1.7, 1.8, 1.9], dtype=np.float32)
        ds_val = np.array([2, 3, 4], dtype=np.int64)
        d_val = np.array(2.5, dtype=np.float32)
        def func(indices, values, dense_shape, default):
            st = tf.SparseTensor(indices, values, dense_shape)
            dense = tf.sparse.to_dense(st, default, validate_indices=True)
            x_ = tf.identity(dense, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: i_val, _INPUT1: v_val, _INPUT2: ds_val, _INPUT3: d_val})

    @check_opset_min_version(11, "Unique")
    def test_sparse_fill_empty_rows(self):
        i_val = np.array([[1, 0, 0], [1, 0, 2], [1, 1, 3], [3, 2, 2], [3, 2, 3]], dtype=np.int64)
        v_val = np.array([1.5, 1.6, 1.7, 1.8, 1.9], dtype=np.float32)
        ds_val = np.array([5, 3, 4], dtype=np.int64)
        d_val = np.array(2.5, dtype=np.float32)
        def func(indices, values, dense_shape, default):
            st = tf.SparseTensor(indices, values, dense_shape)
            st_, indicator = tf.sparse.fill_empty_rows(st, default)
            dense = tf.sparse.to_dense(st_, 0, validate_indices=False)
            dense_ = tf.identity(dense, name=_TFOUTPUT)
            indicator_ = tf.identity(indicator, name=_TFOUTPUT1)
            return dense_, indicator_
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: i_val, _INPUT1: v_val, _INPUT2: ds_val, _INPUT3: d_val})

    @check_opset_min_version(11, "CumSum")
    def test_sparse_reshape(self):
        indices_val = np.array([[1, 0, 0], [1, 0, 2], [1, 1, 3], [3, 2, 2], [3, 2, 3]], dtype=np.int64)
        values_val = np.array([1.5, 1.6, 1.7, 1.8, 1.9], dtype=np.int64)
        dense_shape_val = np.array([5, 3, 4], dtype=np.int64)
        new_shape_val = np.array([2, -1, 1, 3], dtype=np.int64)
        def func(indices, values, dense_shape, new_shape):
            st = tf.SparseTensor(indices, values, dense_shape)
            st_ = tf.sparse.reshape(st, new_shape)
            indices_ = st_.indices
            dense_shape_ = st_.dense_shape
            indices_ = tf.identity(indices_, name=_TFOUTPUT)
            dense_shape_ = tf.identity(dense_shape_, name=_TFOUTPUT1)
            return indices_, dense_shape_
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: indices_val, _INPUT1: values_val,
                                                        _INPUT2: dense_shape_val, _INPUT3: new_shape_val})

    @check_opset_min_version(11, "CumSum")
    def test_sparse_reshape_unknown_rank(self):
        indices_val = np.array([[1, 0, 0], [1, 0, 2], [1, 1, 3], [3, 2, 2], [3, 2, 3]], dtype=np.int64)
        values_val = np.array([1.5, 1.6, 1.7, 1.8, 1.9], dtype=np.int64)
        dense_shape_val = np.array([5, 3, 4], dtype=np.int64)
        new_shape_val = np.array([2, 10, 1, 3], dtype=np.int64)
        shape_pad_val = np.zeros((1, 2), dtype=np.int64)
        def func(indices, dense_shape, new_shape, shape_pad):
            st = tf.SparseTensor(indices, values_val, dense_shape)
            # Some hackery to make the rank unknown
            new_shape_ = tf.pad(new_shape, shape_pad, constant_values=0)
            st_ = tf.sparse.reshape(st, new_shape_)
            indices_ = st_.indices
            dense_shape_ = st_.dense_shape
            indices_ = tf.identity(indices_, name=_TFOUTPUT)
            dense_shape_ = tf.identity(dense_shape_, name=_TFOUTPUT1)
            return indices_, dense_shape_
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: indices_val, _INPUT1: dense_shape_val,
                                                        _INPUT2: new_shape_val, _INPUT3: shape_pad_val})

    @check_tf_min_version("1.14", "ragged needs tf 1.14")
    @check_opset_min_version(11, "CumSum")
    def test_ragged_tensor_to_sparse(self):
        splits_val1 = np.array([0, 1, 1, 5], dtype=np.int32)
        splits_val2 = np.array([0, 3, 3, 5, 9, 10], dtype=np.int32)
        dense_vals_val = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32)
        def func(splits1, splits2, rt_dense_values):
            x = tf.RaggedTensor.from_nested_row_splits(rt_dense_values, [splits1, splits2], validate=True)
            s = x.to_sparse()
            indices, values, shape = s.indices, s.values, s.dense_shape
            indices = tf.identity(indices, name=_TFOUTPUT)
            values = tf.identity(values, name=_TFOUTPUT1)
            shape = tf.identity(shape, name=_TFOUTPUT2)
            return indices, values, shape
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2],
                            {_INPUT: splits_val1, _INPUT1: splits_val2, _INPUT2: dense_vals_val})

    @check_tf_min_version("1.14", "ragged needs tf 1.14")
    @check_opset_min_version(11, "CumSum")
    def test_ragged_gather(self):
        splits_val = np.array([0, 3, 3, 5, 9, 10], dtype=np.int32)
        dense_vals_val = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32)
        indices_val = np.array([1, 3, 2, 0, 1, 1, 4, 3, 3], dtype=np.int32)
        def func(splits, rt_dense_values, indices):
            x = tf.RaggedTensor.from_nested_row_splits(rt_dense_values, [splits], validate=True)
            g = tf.gather(x, indices)
            rt_nested_splits = tf.identity(g.row_splits, name=_TFOUTPUT)
            rt_dense_values = tf.identity(g.flat_values, name=_TFOUTPUT1)
            return rt_nested_splits, rt_dense_values
        self._run_test_case(func, [_OUTPUT, _OUTPUT1],
                            {_INPUT: splits_val, _INPUT1: dense_vals_val, _INPUT2: indices_val})

    @check_tf_min_version("1.14", "ragged needs tf 1.14")
    @check_opset_min_version(11, "CumSum")
    @skip_tflite("unknown rank")
    def test_ragged_tensor_to_tensor(self):
        splits_val1 = np.array([0, 1, 1, 5], dtype=np.int32)
        splits_val2 = np.array([0, 3, 3, 5, 9, 10], dtype=np.int32)
        dense_vals_val = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32)
        def func(splits1, splits2, rt_dense_values):
            x = tf.RaggedTensor.from_nested_row_splits(rt_dense_values, [splits1, splits2], validate=True)
            y = x.to_tensor(default_value=7)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: splits_val1, _INPUT1: splits_val2, _INPUT2: dense_vals_val})

    @check_tf_min_version("1.14", "ragged needs tf 1.14")
    @check_opset_min_version(11, "CumSum")
    @skip_tflite("unknown rank")
    def test_ragged_tensor_to_tensor_row_ids(self):
        ids_val1 = np.array([0, 0, 0, 2, 2], dtype=np.int32)
        ids_val2 = np.array([0, 0, 2, 2, 2, 3, 3, 4], dtype=np.int32)
        dense_vals_val = make_xval([8, 2, 3])
        def func(ids1, ids2, rt_dense_values):
            x = tf.RaggedTensor.from_nested_value_rowids(rt_dense_values, [ids1, ids2], [4, 5])
            y = x.to_tensor(default_value=7)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: ids_val1, _INPUT1: ids_val2, _INPUT2: dense_vals_val})

    @check_tf_min_version("2.2", "ragged to_tensor with constrained shape")
    @check_opset_min_version(11, "CumSum")
    def test_ragged_tensor_to_tensor_constrain_shape(self):
        splits_val1 = np.array([0, 1, 1, 5], dtype=np.int32)
        splits_val2 = np.array([0, 3, 3, 5, 9, 10], dtype=np.int32)
        dense_vals_val = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32)
        dense_vals_val = make_xval([10, 2, 3])
        def func(splits1, splits2, rt_dense_values):
            x = tf.RaggedTensor.from_nested_row_splits(rt_dense_values, [splits1, splits2], validate=True)
            y = x.to_tensor(default_value=7, shape=[20, None, 2, None, 3])
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: splits_val1, _INPUT1: splits_val2, _INPUT2: dense_vals_val})

    @check_tf_min_version("1.14", "ragged needs tf 1.14")
    @check_opset_min_version(11, "Range")
    def test_ragged_range_float(self):
        starts_val = np.array([0, 0, 1, 10, 0.5, 0.5], dtype=np.float32)
        limits_val = np.array([-5, -2, 7, 100, 1, 1], dtype=np.float32)
        deltas_val = np.array([-1, 1, 2, 20, 1, 1.1], dtype=np.float32)
        def func(starts, limits, deltas):
            x = tf.ragged.range(starts, limits, deltas)
            rt_nested_splits = tf.identity(x.row_splits, name=_TFOUTPUT)
            rt_dense_values = tf.identity(x.flat_values, name=_TFOUTPUT1)
            return rt_nested_splits, rt_dense_values
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: starts_val, _INPUT1: limits_val,
                                                        _INPUT2: deltas_val})

    @check_tf_min_version("1.14", "ragged needs tf 1.14")
    @check_opset_min_version(11, "Range")
    def test_ragged_range_int(self):
        starts_val = np.array([0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        limits_val = np.array([-6, -5, -4, -1, 0, 1, 4, 5, 6, 2, -2], dtype=np.int32)
        deltas_val = np.array([-5, -5, -5, -5, 5, 5, 5, 5, 5, 1, -1], dtype=np.int32)
        def func(starts, limits, deltas):
            x = tf.ragged.range(starts, limits, deltas)
            rt_nested_splits = tf.identity(x.row_splits, name=_TFOUTPUT)
            rt_dense_values = tf.identity(x.flat_values, name=_TFOUTPUT1)
            return rt_nested_splits, rt_dense_values
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: starts_val, _INPUT1: limits_val,
                                                        _INPUT2: deltas_val})

    @check_tf_min_version("1.14", "ragged needs tf 1.14")
    @check_opset_min_version(11, "Range")
    def test_ragged_range_scalar(self):
        starts_val = np.array(0, dtype=np.int32)
        limits_val = np.array([5, -1, -1, 2, 7, 100, 4, 5, 6], dtype=np.int32)
        deltas_val = np.array(1, dtype=np.int32)
        def func(starts, limits, deltas):
            x = tf.ragged.range(starts, limits, deltas)
            rt_nested_splits = tf.identity(x.row_splits, name=_TFOUTPUT)
            rt_dense_values = tf.identity(x.flat_values, name=_TFOUTPUT1)
            return rt_nested_splits, rt_dense_values
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: starts_val, _INPUT1: limits_val,
                                                        _INPUT2: deltas_val})

    @check_tf_min_version("2.2", "ragged variant needs tf 2.2")
    @check_opset_min_version(13, "Loop over tensor sequences")
    def test_ragged_to_variant(self):
        splits_val = np.array([0, 3, 3, 5, 9, 10], dtype=np.int32)
        dense_vals_val = np.arange(10 * 3 * 2, dtype=np.float32).reshape([10, 3, 2])

        def fn(elem):
            res = elem + elem * elem
            return res

        def func(splits, rt_dense_values):
            x = tf.RaggedTensor.from_nested_row_splits(rt_dense_values, [splits], validate=True)
            y = tf.map_fn(fn, x)
            return tf.identity(y.row_splits, name=_TFOUTPUT), tf.identity(y.flat_values, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: splits_val, _INPUT1: dense_vals_val})

    @check_tf_min_version("2.2", "ragged variant needs tf 2.2")
    @check_opset_min_version(13, "Loop over tensor sequences")
    def test_ragged_to_variant_unknown_shape(self):
        splits_val = np.array([0, 3, 3, 5, 9, 10], dtype=np.int64)
        dense_vals_shape = np.array([10, 3, 2], dtype=np.int32)
        splits_pads_val = np.array([[0, 0]], dtype=np.int32)

        def fn(elem):
            res = elem + elem * elem
            return res

        def func(splits, rt_dense_values_shape, splits_pads):
            rt_dense_values = tf.ones(rt_dense_values_shape, dtype=tf.int32)
            splits = tf.pad(splits, splits_pads)
            x = tf.RaggedTensor.from_nested_row_splits(rt_dense_values, [splits], validate=True)
            y = tf.map_fn(fn, x)
            return tf.identity(y.row_splits, name=_TFOUTPUT), tf.identity(y.flat_values, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1],
                            {_INPUT: splits_val, _INPUT1: dense_vals_shape, _INPUT2: splits_pads_val})

    @check_opset_min_version(9, "Compress")
    def test_dynamic_partition_both_vector(self):
        data_val = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        part_val = np.array([0, 0, 1, 1, 0, 2, 1, 0], dtype=np.int32)
        def func(data, partitions):
            p1, p2, p3 = tf.dynamic_partition(data, partitions, num_partitions=3)
            p1_ = tf.identity(p1, name=_TFOUTPUT)
            p2_ = tf.identity(p2, name=_TFOUTPUT1)
            p3_ = tf.identity(p3, name=_TFOUTPUT2)
            return p1_, p2_, p3_
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: data_val, _INPUT1: part_val})

    @check_opset_min_version(9, "Compress")
    def test_dynamic_partition_data_tensor(self):
        data_val = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
        part_val = np.array([0, 2, 1, 0, 1], dtype=np.int32)
        def func(data, partitions):
            p1, p2, p3 = tf.dynamic_partition(data, partitions, num_partitions=3)
            p1_ = tf.identity(p1, name=_TFOUTPUT)
            p2_ = tf.identity(p2, name=_TFOUTPUT1)
            p3_ = tf.identity(p3, name=_TFOUTPUT2)
            return p1_, p2_, p3_
        self._run_test_case(func, [_OUTPUT, _OUTPUT1, _OUTPUT2], {_INPUT: data_val, _INPUT1: part_val})

    @check_opset_min_version(11, "ScatterElements")
    @unittest.skip("this test is failing for some opsets, disabled until fixed")
    def test_dynamic_stitch_both_vector(self):
        data_val = np.array([[5, 1, 3], [7, 2, 4]], dtype=np.float32)
        indices_val = np.array([[0, 1, 4], [2, 3, 5]], dtype=np.int32)
        def func(indices, data):
            x = tf.dynamic_stitch(tf.unstack(indices), tf.unstack(data))
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: indices_val, _INPUT1: data_val})

    @check_opset_min_version(11, "ScatterElements")
    def test_dynamic_stitch_data_tensor(self):
        data_val = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape((2, 3, 2, 4))
        indices_val = np.array([[0, 1, 4], [2, 3, 5]], dtype=np.int32)
        def func(indices, data):
            x = tf.dynamic_stitch(tf.unstack(indices), tf.unstack(data))
            x_ = tf.identity(x, name=_TFOUTPUT)
            return x_
        self._run_test_case(func, [_OUTPUT], {_INPUT: indices_val, _INPUT1: data_val})

    @check_opset_min_version(10, "Conv2DBackpropInput")
    def test_Conv2DBackpropInput_const(self):
        input_sizes_val_ = np.array([1, 10, 10, 3], dtype=np.int32)
        def func(filter_val, out_backprop_val):
            input_sizes_val = tf.constant(input_sizes_val_, dtype=tf.int32)
            return conv2d_backprop_input(input_sizes=input_sizes_val, filter=filter_val,
                                         out_backprop=out_backprop_val, strides=[1, 1, 1, 1],
                                         padding='SAME', name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 10, 10, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: filters_val, _INPUT1: out_backprop_val})

    @check_opset_min_version(10, "Conv2DBackpropInput")
    def test_Conv2DBackpropInput_const_strided(self):
        input_sizes_val_ = np.array([1, 10, 10, 3], dtype=np.int32)
        def func(filter_val, out_backprop_val):
            input_sizes_val = tf.constant(input_sizes_val_, dtype=tf.int32)
            return conv2d_backprop_input(input_sizes=input_sizes_val, filter=filter_val,
                                         out_backprop=out_backprop_val, strides=[1, 2, 2, 1],
                                         padding='SAME', name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 5, 5, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: filters_val, _INPUT1: out_backprop_val})

    @check_tf_min_version("1.15", "tf.repeat needs tf 1.15")
    @check_opset_min_version(10, "Conv2DBackpropInput")
    def test_Conv2DBackpropInput_shape_implied(self):
        batch_dim_val = np.array(1, dtype=np.int32)
        def func(filter_val, out_backprop_val, batch_dim):
            out_backprop_val = tf.repeat(out_backprop_val, batch_dim, axis=0)
            s = tf.shape(out_backprop_val)
            t1 = tf.constant([0], dtype=tf.int32)
            t2 = tf.constant([1], dtype=tf.int32)
            batch_dim = tf.strided_slice(s, t1, t2, shrink_axis_mask=1)
            # Sometimes the size given is a stack of constants with unknown batch dim
            input_sizes_val = tf.stack([batch_dim, 10, 10, 3])
            return conv2d_backprop_input(input_sizes=input_sizes_val, filter=filter_val,
                                         out_backprop=out_backprop_val, strides=[1, 2, 2, 1],
                                         padding='SAME', name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 5, 5, 5]).astype(np.float32)
        def graph_validator(g):
            for n in g.get_nodes():
                if n.type == 'ConvTranspose':
                    return "pads" in n.attr or "output_shape" in n.attr
            return False
        self._run_test_case(func, [_OUTPUT], {_INPUT: filters_val, _INPUT1: out_backprop_val, _INPUT2: batch_dim_val},
                            graph_validator=graph_validator)

    @check_opset_min_version(10, "Conv2DBackpropInput")
    def test_Conv2DBackpropInput_const_valid(self):
        input_sizes_val_ = np.array([1, 12, 12, 3], dtype=np.int32)
        def func(filter_val, out_backprop_val):
            input_sizes_val = tf.constant(input_sizes_val_, dtype=tf.int32)
            return conv2d_backprop_input(input_sizes=input_sizes_val, filter=filter_val,
                                         out_backprop=out_backprop_val, strides=[1, 1, 1, 1],
                                         padding='VALID', name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 10, 10, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: filters_val, _INPUT1: out_backprop_val})

    @check_opset_min_version(10, "Conv2DBackpropInput")
    def test_Conv2DBackpropInput(self):
        def func(input_sizes, filters, out_backprop):
            return conv2d_backprop_input(input_sizes, filters, out_backprop, strides=[1, 1, 1, 1],
                                         padding='SAME', name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 10, 10, 5]).astype(np.float32)
        input_sizes_val = np.array([1, 10, 10, 3], dtype=np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_sizes_val, _INPUT1: filters_val, _INPUT2: out_backprop_val})

    @check_opset_min_version(10, "Conv2DBackpropInput")
    def test_Conv2DBackpropInput_strided(self):
        def func(input_sizes, filters, out_backprop):
            return conv2d_backprop_input(input_sizes, filters, out_backprop, strides=[1, 2, 2, 1], padding='SAME',
                                         name=_TFOUTPUT)
        input_sizes_val = np.array([1, 10, 10, 3], dtype=np.int32)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 5, 5, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_sizes_val, _INPUT1: filters_val, _INPUT2: out_backprop_val})

    @check_opset_min_version(10, "Conv2DBackpropInput")
    def test_Conv2DBackpropInput_valid(self):
        def func(input_sizes, filters, out_backprop):
            return conv2d_backprop_input(input_sizes, filters, out_backprop, strides=[1, 1, 1, 1],
                                         padding='VALID', name=_TFOUTPUT)
        input_sizes_val = np.array([1, 12, 12, 3], dtype=np.int32)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 10, 10, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_sizes_val, _INPUT1: filters_val, _INPUT2: out_backprop_val})

    @check_opset_min_version(12, "Conv2DBackpropInput with strided workaround")
    def test_Conv2DBackpropInput_strided_same(self):
        def func(input_sizes, filters, out_backprop):
            return conv2d_backprop_input(input_sizes, filters, out_backprop, strides=[1, 5, 10, 1], padding='SAME',
                                         name=_TFOUTPUT)
        input_sizes_val = np.array([1, 10, 10, 3], dtype=np.int32)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 5]).astype(np.float32)
        out_backprop_val = np.random.randint(low=0, high=256, size=[1, 2, 1, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: input_sizes_val, _INPUT1: filters_val, _INPUT2: out_backprop_val})

    @check_opset_min_version(10, "Conv3DBackpropInputV2")
    def test_Conv3DBackpropInputV2_const(self):
        output_shape_val_ = np.array([1, 10, 10, 10, 3], dtype=np.int32)
        def func(value, filters):
            output_shape_val = tf.constant(output_shape_val_, dtype=tf.int32)
            return conv3d_transpose(value, filters, output_shape_val, strides=[1, 1, 1, 1, 1],
                                    padding='SAME', data_format="NDHWC", name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 3, 5]).astype(np.float32)
        value_val = np.random.randint(low=0, high=256, size=[1, 10, 10, 10, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: value_val, _INPUT1: filters_val}, rtol=1e-6)

    @check_opset_min_version(10, "Conv3DBackpropInputV2")
    def test_Conv3DBackpropInputV2_const_strided(self):
        output_shape_val_ = np.array([1, 10, 10, 10, 3], dtype=np.int32)
        def func(value, filters):
            output_shape_val = tf.constant(output_shape_val_, dtype=tf.int32)
            return conv3d_transpose(value, filters, output_shape_val, strides=[1, 2, 2, 2, 1],
                                    padding='SAME', data_format="NDHWC", name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 3, 5]).astype(np.float32)
        value_val = np.random.randint(low=0, high=256, size=[1, 5, 5, 5, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: value_val, _INPUT1: filters_val}, rtol=1e-6)

    @check_opset_min_version(10, "Conv3DBackpropInputV2")
    def test_Conv3DBackpropInputV2_const_valid(self):
        output_shape_val_ = np.array([1, 12, 12, 12, 3], dtype=np.int32)
        def func(value, filters):
            output_shape_val = tf.constant(output_shape_val_, dtype=tf.int32)
            return conv3d_transpose(value, filters, output_shape_val, strides=[1, 1, 1, 1, 1],
                                    padding='VALID', data_format="NDHWC", name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 3, 5]).astype(np.float32)
        value_val = np.random.randint(low=0, high=256, size=[1, 10, 10, 10, 5]).astype(np.float32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: value_val, _INPUT1: filters_val}, rtol=1e-6)

    @check_opset_min_version(10, "Conv3DBackpropInputV2")
    def test_Conv3DBackpropInputV2(self):
        def func(value, filters, output_shape):
            return conv3d_transpose(value, filters, output_shape, strides=[1, 1, 1, 1, 1],
                                    padding='SAME', data_format="NDHWC", name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[2, 3, 4, 4, 5]).astype(np.float32)
        value_val = np.random.randint(low=0, high=256, size=[2, 7, 8, 9, 5]).astype(np.float32)
        output_shape_val = np.array([2, 7, 8, 9, 4], dtype=np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: value_val, _INPUT1: filters_val, _INPUT2: output_shape_val},
                            rtol=1e-6)

    @check_opset_min_version(10, "Conv3DBackpropInputV2")
    def test_Conv3DBackpropInputV2_strided(self):
        def func(value, filters, output_shape):
            return conv3d_transpose(value, filters, output_shape, strides=[1, 2, 2, 2, 1],
                                    padding='SAME', data_format="NDHWC", name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 3, 5]).astype(np.float32)
        value_val = np.random.randint(low=0, high=256, size=[1, 5, 5, 5, 5]).astype(np.float32)
        output_shape_val = np.array([1, 10, 10, 10, 3], dtype=np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: value_val, _INPUT1: filters_val, _INPUT2: output_shape_val},
                            rtol=1e-6)

    @check_opset_min_version(10, "Conv3DBackpropInputV2")
    def test_Conv3DBackpropInputV2_valid(self):
        def func(value, filters, output_shape):
            return conv3d_transpose(value, filters, output_shape, strides=[1, 1, 1, 1, 1],
                                    padding='VALID', data_format="NDHWC", name=_TFOUTPUT)
        filters_val = np.random.randint(low=0, high=256, size=[3, 3, 3, 3, 5]).astype(np.float32)
        value_val = np.random.randint(low=0, high=256, size=[1, 10, 10, 10, 5]).astype(np.float32)
        output_shape_val = np.array([1, 12, 12, 12, 3], dtype=np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: value_val, _INPUT1: filters_val, _INPUT2: output_shape_val},
                            rtol=1e-6)

    @check_opset_min_version(12, "Conv3DBackpropInputV2 with strided workaround")
    def test_Conv3DBackpropInputV2_strided_same(self):
        def func(value, filters, output_shape):
            return conv3d_transpose(value, filters, output_shape, strides=[1, 10, 4, 3, 1],
                                    padding='SAME', data_format="NDHWC", name=_TFOUTPUT)
        filters_val = np.random.randint(low=1, high=256, size=[1, 1, 1, 1, 1]).astype(np.float32)
        value_val = np.random.randint(low=1, high=256, size=[1, 3, 2, 5, 1]).astype(np.float32)
        output_shape_val = np.array([1, 30, 8, 15, 1], dtype=np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: value_val, _INPUT1: filters_val, _INPUT2: output_shape_val},
                            rtol=1e-6)

    @check_opset_min_version(8, "CategoryMapper")
    @skip_tfjs("TFJS does not initialize table")
    @skip_onnx_checker("ONNX can't do type inference on CategoryMapper")
    def test_hashtable_lookup(self):
        filnm = "vocab.tmp"
        words = ["apple", "pear", "banana", "cherry", "grape"]
        query = np.array(['cherry'], dtype=object)
        with open(filnm, "w") as f:
            for word in words:
                f.write(word + "\n")
        def func(query_holder):
            hash_table = lookup_ops.index_table_from_file(filnm)
            lookup_results = hash_table.lookup(query_holder)
            ret = tf.add(lookup_results, 0, name=_TFOUTPUT)
            return ret
        self._run_test_case(func, [_OUTPUT], {_INPUT: query}, as_session=True)
        os.remove(filnm)

    @check_opset_min_version(8, "CategoryMapper")
    @skip_tfjs("TFJS does not initialize table")
    @skip_onnx_checker("ONNX can't do type inference on CategoryMapper")
    def test_hashtable_lookup_invert(self):
        filnm = "vocab.tmp"
        words = ["apple", "pear", "banana", "cherry", "grape"]
        query = np.array([3], dtype=np.int64)
        with open(filnm, "w") as f:
            for word in words:
                f.write(word + "\n")
        def func(query_holder):
            hash_table = lookup_ops.index_to_string_table_from_file(filnm)
            lookup_results = hash_table.lookup(query_holder)
            ret = tf.identity(lookup_results, name=_TFOUTPUT)
            return ret
        self._run_test_case(func, [_OUTPUT], {_INPUT: query}, as_session=True)
        os.remove(filnm)

    @check_opset_min_version(8, "CategoryMapper")
    @skip_tfjs("TFJS does not initialize table")
    def test_hashtable_lookup_const(self):
        filnm = "vocab.tmp"
        words = ["apple", "pear", "banana", "cherry ♥", "grape"]
        query_val = np.array(['cherry ♥', 'banana'], dtype=object).reshape((1, 2, 1))
        with open(filnm, "w", encoding='UTF-8') as f:
            for word in words:
                f.write(word + "\n")
        def func():
            hash_table = lookup_ops.index_table_from_file(filnm)
            query = tf.constant(query_val)
            lookup_results = hash_table.lookup(query)
            ret = tf.add(lookup_results, 0, name=_TFOUTPUT)
            return ret
        self._run_test_case(func, [_OUTPUT], {}, as_session=True)
        os.remove(filnm)

    @check_opset_min_version(8, "CategoryMapper")
    @skip_tfjs("TFJS does not initialize table")
    def test_hashtable_lookup_invert_const(self):
        filnm = "vocab.tmp"
        words = ["apple", "pear", "banana", "cherry", "grape"]
        query_val = np.array([3, 2], dtype=np.int64).reshape((1, 2, 1))
        with open(filnm, "w", encoding='UTF-8') as f:
            for word in words:
                f.write(word + "\n")
        def func():
            hash_table = lookup_ops.index_to_string_table_from_file(filnm)
            query = tf.constant(query_val)
            lookup_results = hash_table.lookup(query)
            ret = tf.identity(lookup_results, name=_TFOUTPUT)
            return ret
        self._run_test_case(func, [_OUTPUT], {}, as_session=True)
        os.remove(filnm)

    @skip_tfjs("TFJS does not initialize table")
    def test_hashtable_size(self):
        filnm = "vocab.tmp"
        words = ["apple", "pear", "banana", "cherry", "grape"]
        query = np.array(['cherry'], dtype=object)
        with open(filnm, "w") as f:
            for word in words:
                f.write(word + "\n")
        def func(query_holder):
            hash_table = lookup_ops.index_table_from_file(filnm)
            lookup_size = hash_table.size()
            ret = tf.add(lookup_size, 0, name=_TFOUTPUT)
            return ret
        self._run_test_case(func, [_OUTPUT], {_INPUT: query}, as_session=True)
        os.remove(filnm)

    @check_opset_min_version(11)
    @skip_onnx_checker("Fails. Fix later.")
    def test_matrix_diag_part(self):
        input_vals = [
            np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]], dtype=np.int64),
            np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]], dtype=np.int64),
            np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]],
                      [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]]], dtype=np.int64)]

        def func(input_holder):
            return matrix_diag_part(input_holder, name=_TFOUTPUT)

        for input_val in input_vals:
            self._run_test_case(func, [_OUTPUT], {_INPUT: input_val})

    @check_opset_min_version(8)
    def test_broadcast(self):
        input_tensor_val = np.random.randint(low=0, high=256, size=[2, 3]).astype(np.float32)
        new_shape_val = np.array([3, 2, 3]).astype(np.int64)

        def func(input_tensor, new_shape):
            return tf.broadcast_to(input_tensor, new_shape, _TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: input_tensor_val, _INPUT1: new_shape_val})

    def test_bfloat(self):
        x_val = np.array([0, 1, 2], dtype=np.float32)
        y_val = np.array([3, 4, 5], dtype=np.float32)
        def func(x, y):
            x_ = tf.cast(x, tf.bfloat16)
            y_ = tf.cast(y, tf.bfloat16)
            s_ = tf.add(x_, y_)
            return tf.cast(s_, tf.float32, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(11)
    @check_tf_min_version("2.2")
    @skip_onnx_checker("Fails. Fix later.")
    def test_matrix_diag_part_v3(self):

        def func(X, K):
            v2 = tf.raw_ops.MatrixDiagPartV2(input=X, k=K, padding_value=0.123, name=_TFOUTPUT)
            v3 = tf.raw_ops.MatrixDiagPartV3(input=X, k=K, padding_value=0.123, align='LEFT_RIGHT', name=_TFOUTPUT1)
            return v2, v3

        for x_shape in ([4, 5], [2, 3, 4, 5], [5, 4], [7, 5]):
            x_val = np.random.random(x_shape).astype(np.float32)
            for raw_k in ([0], [1], [3], [-1], [-3], [1, 2], [-2, -1], [-1, 1]):
                k_val = np.array(raw_k).astype(np.int32)
                self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val, _INPUT1: k_val})

    @test_ms_domain()
    def test_inverse(self, extra_opset):
        # this depends on onnx Inverse which was removed from opset-12 but does exists in the ms-domain
        x_val = np.random.random([5, 5]).astype(np.float32)
        def func(x):
            return tf.linalg.inv(x, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, process_args={"extra_opset": [extra_opset]})

    @check_opset_min_version(12)
    def test_squared_distance(self):
        x_val = np.random.random([4, 5]).astype(np.float32)
        y_val = np.random.random([4, 5]).astype(np.float32)
        def func(x, y):
            return tf.math.squared_difference(x, y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.1")
    def test_einsum(self):
        x_val = np.random.random([10]).astype(np.float32)
        y_val = np.random.random([10]).astype(np.float32)
        def func(x, y):
            ret = tf.einsum("i,j->ij", x, y)
            return tf.identity(ret, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.1")
    def test_einsum_to_matmul(self):
        x_val = np.random.random([4, 10, 20]).astype(np.float32)
        y_val = np.random.random([20, 30]).astype(np.float32)
        def func(x, y):
            ret = tf.einsum("bik,kj->bij", x, y)
            return tf.identity(ret, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.1")
    def test_einsum_to_matmul_transpose(self):
        x_val = np.random.random([4, 10, 20]).astype(np.float32)
        y_val = np.random.random([30, 20]).astype(np.float32)
        def func(x, y):
            ret = tf.einsum("bik,jk->bij", x, y)
            return tf.identity(ret, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(7)
    def test_compare(self):
        x_val = np.random.random([10, 20]).astype(np.float32)
        y_val = np.random.random([10, 20]).astype(np.float32)
        def func(x, y):
            return tf.math.less_equal(x, y, name=_TFOUTPUT), \
                   tf.math.greater_equal(x, y, name=_TFOUTPUT1)
        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: x_val, _INPUT1: y_val})

    @check_tf_min_version("1.14", "required for tf.math.is_finite")
    @check_opset_min_version(10)
    def test_is_finite(self):
        x_val = np.array([5.0, 4.8, 6.8, np.inf, np.nan], dtype=np.float32)
        def func(x):
            y = tf.math.is_finite(x)
            return tf.identity(y, name=_TFOUTPUT)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.2")
    def test_matrix_diag_v3_multi_dim(self):
        raw_diag = [[[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]],
                    [[10.0, 11.0, 12.0],
                     [13.0, 14.0, 15.0],
                     [16.0, 17.0, 18.0]]]
        diag_val = np.array(raw_diag).astype(np.float32)
        k_val = np.array([-1, 1]).astype(np.int32)
        row_val = np.array(-1).astype(np.int32)
        col_val = np.array(-1).astype(np.int32)

        def func(diag, k, row, col):
            return tf.raw_ops.MatrixDiagV3(diagonal=diag, k=k, num_rows=row, num_cols=col,
                                           padding_value=0.123, align='RIGHT_RIGHT', name=_TFOUTPUT), \
                   tf.raw_ops.MatrixDiagV2(diagonal=diag, k=k, num_rows=row, num_cols=col,
                                           padding_value=0.123, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1], {_INPUT: diag_val, _INPUT1: k_val,
                                                        _INPUT2: row_val, _INPUT3: col_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.2")
    def test_matrix_diag_v3_multi_dim_min_row(self):
        raw_diag = [[[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0],
                     [10.0, 11.0, 12.0]]]
        diag_val = np.array(raw_diag).astype(np.float32)
        k_val = np.array([2, 3]).astype(np.int32)
        row_val = np.array(-1).astype(np.int32)
        col_val = np.array(6).astype(np.int32)

        def func(diag, k, row, col):
            return tf.raw_ops.MatrixDiagV3(diagonal=diag, k=k, num_rows=row, num_cols=col,
                                           padding_value=0.456, align='LEFT_LEFT', name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: diag_val, _INPUT1: k_val,
                                              _INPUT2: row_val, _INPUT3: col_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.2")
    def test_matrix_diag_v3_single_dim_min_col(self):
        raw_diag = [1.0, 2.0, 3.0]
        diag_val = np.array(raw_diag).astype(np.float32)
        k_val = np.array(-1).astype(np.int32)
        row_val = np.array(5).astype(np.int32)
        col_val = np.array(-1).astype(np.int32)

        def func(diag, k, row, col):
            return tf.raw_ops.MatrixDiagV3(diagonal=diag, k=k, num_rows=row, num_cols=col,
                                           padding_value=0.789, align='LEFT_RIGHT', name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: diag_val, _INPUT1: k_val,
                                              _INPUT2: row_val, _INPUT3: col_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.2")
    def test_matrix_diag_v3_2single_dim_row_col(self):
        raw_diag = [[1, 2, 3], [4, 5, 6]]
        diag_val = np.array(raw_diag).astype(np.int64)
        k_val = np.array(0).astype(np.int32)
        row_val = np.array(3).astype(np.int32)
        col_val = np.array(4).astype(np.int32)

        def func(diag, k, row, col):
            return tf.raw_ops.MatrixDiagV3(diagonal=diag, k=k, num_rows=row, num_cols=col,
                                           padding_value=7, align='LEFT_RIGHT', name=_TFOUTPUT), \
                   tf.raw_ops.MatrixDiag(diagonal=diag, name=_TFOUTPUT1)

        self._run_test_case(func, [_OUTPUT, _OUTPUT1],
                            {_INPUT: diag_val, _INPUT1: k_val,
                             _INPUT2: row_val, _INPUT3: col_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.2")
    def test_matrix_diag_v3_1single_dim_row_col(self):
        raw_diag = [1, 2, 3, 4, 5]
        diag_val = np.array(raw_diag).astype(np.int64)
        k_val = np.array(0).astype(np.int32)
        row_val = np.array(5).astype(np.int32)
        col_val = np.array(10).astype(np.int32)

        def func(diag, k, row, col):
            return tf.raw_ops.MatrixDiagV3(diagonal=diag, k=k, num_rows=row, num_cols=col,
                                           padding_value=7, align='LEFT_RIGHT', name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: diag_val, _INPUT1: k_val,
                                              _INPUT2: row_val, _INPUT3: col_val})

    @check_opset_min_version(12)
    @check_tf_min_version("2.2")
    @skip_onnx_checker("Checker fails. Fix later.")
    def test_matrix_set_diag_v3(self):
        input_val = np.array([[[7, 7, 7, 7],
                               [7, 7, 7, 7],
                               [7, 7, 7, 7]],
                              [[7, 7, 7, 7],
                               [7, 7, 7, 7],
                               [7, 7, 7, 7]]]).astype(np.int64)
        diag_val = np.array([[1, 2, 3],
                             [4, 5, 6]]).astype(np.int64)
        k_val = np.array([0]).astype(np.int32)

        def func(base_matrix, diag, k):
            return tf.raw_ops.MatrixSetDiagV3(input=base_matrix, diagonal=diag, k=k, align='RIGHT_LEFT', name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: input_val, _INPUT1: diag_val, _INPUT2: k_val})

    @check_opset_min_version(10)
    @check_tf_min_version("1.14")
    @skip_tfjs("TFJS executes model incorrectly")
    def test_fakequant_with_min_max(self):
        def func(x):
            ret = fake_quant_with_min_max_args(
                x, min=-1024, max=1023, num_bits=8, narrow_range=False, name=None)
            return tf.identity(ret, name=_TFOUTPUT)

        x_val = np.random.random(size=[4, 3]).astype(np.float32) * 2048. - 1024.
        x_val0 = np.abs(x_val)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val0}, rtol=1e-6, atol=1e-4)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-6, atol=1e-4)

        x_val = np.random.random(size=[4, 3]).astype(np.float32) * 2048. - 1024.
        x_val[0, 0] = -1024
        x_val[0, 1] = -1023
        x_val[0, 2] = 1024
        x_val[1, 0] = 1023
        x_val[1, 1] = 1025
        x_val[1, 2] = -1025
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-6, atol=1e-4)

    @check_opset_min_version(10)
    @check_tf_min_version("1.14")
    def test_fakequant_with_min_max_same_sign(self):
        def func_neg(x):
            ret = fake_quant_with_min_max_args(
                x, min=-1024*3, max=-1024, num_bits=8, narrow_range=False, name=None)
            return tf.identity(ret, name=_TFOUTPUT)

        x_val = np.random.random(size=[4, 3]).astype(np.float32) * 2048. - 1024 * 3.
        try:
            self._run_test_case(func_neg, [_OUTPUT], {_INPUT: x_val}, rtol=1e-6, atol=1e-4)
        except ValueError:
            pass

    @check_opset_min_version(10)
    @check_tf_min_version("1.14")
    @skip_tfjs("Results differ slightly in TFJS")
    def test_fakequant_with_min_max_vars(self):
        def func(x):
            ret = fake_quant_with_min_max_vars(
                x, min=-1024, max=1023, num_bits=8, narrow_range=False, name=None)
            return tf.identity(ret, name=_TFOUTPUT)

        x_val = np.random.random(size=[4, 3]).astype(np.float32) * 2048. - 1024.
        x_val0 = np.abs(x_val)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val0}, rtol=1e-6, atol=1e-4)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-6, atol=1e-4)

        x_val = np.random.random(size=[4, 3]).astype(np.float32) * 2048. - 1024.
        x_val[0, 0] = -1024
        x_val[0, 1] = -1023
        x_val[0, 2] = 1024
        x_val[1, 0] = 1023
        x_val[1, 1] = 1025
        x_val[1, 2] = -1025
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, rtol=1e-6, atol=1e-4)

    @check_opset_min_version(9, "atan2")
    def test_atan2(self):
        # Test all possible pairs of pos, neg, zero for x and y.

        def atan2(y, x):
            sx = np.sign(x)
            sy = np.sign(y)
            pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-np.pi/2)
            atan_part = np.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
            return atan_part + pi_part

        test_pairs = [[y, x] for x in [3., -4., 0.] for y in [5., -6., 0.]]
        y_val = np.array([y for y, x in test_pairs], dtype=np.float32)
        x_val = np.array([x for y, x in test_pairs], dtype=np.float32)
        assert_almost_equal(np.arctan2(y_val, x_val), atan2(y_val, x_val))

        def func(y, x):
            atan2_ = tf.math.atan2(y, x)
            return tf.identity(atan2_, name=_TFOUTPUT)

        self._run_test_case(
            func, [_OUTPUT], {_INPUT: y_val, _INPUT2: x_val}, rtol=1e-06)

    def _conv_kernel_as_input_test(self, x_val, w_val, strides=None,
                                   padding="VALID", dilations=None, rtol=1e-07):
        if strides is None:
            strides = _STRIDE1x1
        if dilations is None:
            dilations = _DILATIONS1x1

        def func(x, kernel):
            conv = tf.nn.conv2d(x, kernel, strides=strides, padding=padding,
                                dilations=dilations)
            return tf.identity(conv, name=_TFOUTPUT)

        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT2: w_val}, rtol=rtol)

    def test_conv2d_1_kernel_as_input(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w_val = np.array([[2., 1., 1.],
                          [1., 3., 1.],
                          [1., 1., 4.]], dtype=np.float32).reshape(_KERNEL3x3)
        self._conv_kernel_as_input_test(x_val, w_val)

    def test_equal_with_different_parameters(self):
        input_val = np.array([5], dtype=np.int32)

        def func(input_val):
            tensor = tf.zeros(input_val)
            input_size = tf.size(tensor)
            constant = tf.constant(3, dtype=tf.int32)
            return tf.math.equal(input_size, constant, name="output")

        feed_dict = {"input:0": input_val}
        input_names_with_port = ["input:0"]
        output_names_with_port = ["output:0"]

        current_opset = self.config.opset
        self.config.opset = 12
        try:
            self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)
        finally:
            self.config.opset = current_opset

    @check_tf_min_version("1.14")
    @skip_tf_versions("2.4", "Fails to run on TF 2.4.x")
    @skip_tfjs("Fails to run tfjs model")
    def test_rfft_ops(self):

        def dft_slow(x, M, fft_length):
            xt = x[:, :fft_length].T
            size = fft_length // 2 + 1
            res = np.dot(M[:, :, :fft_length], xt)[:, :size, :]
            return np.transpose(res, (0, 2, 1))

        x_val = make_xval([2, 4]).astype(np.float32)
        M_both = make_dft_constant(x_val.shape[1], x_val.dtype, x_val.shape[1])
        fft = dft_slow(x_val, M_both, x_val.shape[1])
        fft_npy = np.fft.rfft(x_val)
        assert_almost_equal(fft[0, :, :], np.real(fft_npy))
        assert_almost_equal(fft[1, :, :], np.imag(fft_npy))

        x_val = make_xval([2, 4]).astype(np.float32)
        M_both = make_dft_constant(x_val.shape[1], x_val.dtype, x_val.shape[1]-1)
        fft = dft_slow(x_val, M_both, x_val.shape[1]-1)
        fft_npy = np.fft.rfft(x_val, x_val.shape[1]-1)
        assert_almost_equal(fft[0, :, :], np.real(fft_npy))
        assert_almost_equal(fft[1, :, :], np.imag(fft_npy))

        x_val = make_xval([3, 4]).astype(np.float32)
        def func1(x):
            op_ = tf.signal.rfft(x)
            return tf.abs(op_, name=_TFOUTPUT)
        self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val})

        def func2(x):
            op_ = tf.signal.rfft(x)
            return tf.cos(op_, name=_TFOUTPUT)
        with self.assertRaises(ValueError):
            self._run_test_case(func2, [_OUTPUT], {_INPUT: x_val})

        def func3(x):
            op_ = tf.signal.rfft(x)
            return tf.identity(op_, name=_TFOUTPUT)
        with self.assertRaises(ValueError):
            self._run_test_case(func3, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @skip_tf_versions("2.4", "Fails to run on TF 2.4.x")
    @skip_tfjs("TFJS executes rfft with poor accuracy")
    @check_opset_min_version(10, "Slice")
    def test_rfft_ops_fft_length(self):

        x_val = make_xval([3, 9]).astype(np.float32)
        def func1_length(x):
            op_ = tf.signal.rfft(x, np.array([8], dtype=np.int32))
            return tf.abs(op_, name=_TFOUTPUT)
        self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @skip_tf_versions("2.4", "Fails to run on TF 2.4.x")
    @skip_tfjs("TFJS executes rfft with poor accuracy")
    @check_opset_min_version(10, "Slice")
    def test_rfft_ops_fft_length_many(self):
        for i in range(4, 7):
            for j in range(4, 7):
                for m in range(0, 3):
                    with self.subTest(shape=(i, j), fft_length=j-m):
                        x_val = make_xval([i, j]).astype(np.float32)
                        def func1_length(x):
                            op_ = tf.signal.rfft(x, np.array([j-m], dtype=np.int32))
                            return tf.abs(op_, name=_TFOUTPUT)
                        self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @skip_tf_versions("2.4", "Fails to run on TF 2.4.x")
    @check_opset_min_version(10, "Slice")
    def test_rfft_ops_fft_length_many_bigger(self):
        for i in range(4, 7):
            for j in range(4, 7):
                for m in range(0, 3):
                    with self.subTest(shape=(i, j), fft_length=j+m):
                        x_val = make_xval([i, j]).astype(np.float32) / 10
                        def func1_length(x):
                            op_ = tf.signal.rfft(x, np.array([j+m], dtype=np.int32))
                            return tf.abs(op_, name=_TFOUTPUT)
                        self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @skip_tflite("Slight accuracy issues with some shapes")
    @skip_tfjs("TFJS executes rfft with poor accuracy")
    @check_opset_min_version(10, "Slice")
    def test_rfft_ops_fft_length_many_larger(self):
        for i in range(4, 7):
            for j in range(4, 7):
                for m in range(-3, 3):
                    with self.subTest(shape=(3, i, j), fft_length=j+m):
                        x_val = make_xval([3, i, j]).astype(np.float32) / 10
                        def func1_length(x):
                            op_ = tf.signal.rfft(x, np.array([j+m], dtype=np.int32))
                            return tf.abs(op_, name=_TFOUTPUT)
                        self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val}, optimize=False)
                        self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "CumSum")
    def test_rfft2d_ops(self):

        x_val = make_xval([3, 4]).astype(np.float32)

        def func1(x):
            op_ = tf.signal.rfft2d(x)
            return tf.abs(op_, name=_TFOUTPUT)
        self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val}, optimize=False)
        self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val})

        def func2(x):
            op_ = tf.signal.rfft2d(x)
            return tf.cos(op_, name=_TFOUTPUT)
        with self.assertRaises(ValueError):
            self._run_test_case(func2, [_OUTPUT], {_INPUT: x_val})

        def func3(x):
            op_ = tf.signal.rfft2d(x)
            return tf.identity(op_, name=_TFOUTPUT)
        with self.assertRaises(ValueError):
            self._run_test_case(func3, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "CumSum")
    def test_rfft2d_ops_fft_length(self):

        x_val = make_xval([3, 4]).astype(np.float32)
        def func1_length(x):
            op_ = tf.signal.rfft2d(x, np.array([3, 3], dtype=np.int32))
            return tf.abs(op_, name=_TFOUTPUT)
        with self.subTest(optimize=False):
            self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val}, optimize=False)
        with self.subTest(optimize=True):
            self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "CumSum")
    def test_rfft2d_ops_fft_length_many(self):
        for i in range(7, 4, -1):
            for j in range(7, 4, -1):
                for m in range(0, 3):
                    for n in range(0, 3):
                        for opt in [False, True]:
                            with self.subTest(shape=(i, j), fft_length=(m, n), optimize=opt):
                                x_val = make_xval([i, j]).astype(np.float32) / 100
                                def func1_length(x):
                                    op_ = tf.signal.rfft2d(x, np.array([i-m, j-n], dtype=np.int32))
                                    return tf.abs(op_, name=_TFOUTPUT)
                                self._run_test_case(func1_length, [_OUTPUT], {_INPUT: x_val}, optimize=opt)

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "CumSum")
    @unittest.skipIf(True, reason="Not fully implemented for dynamic shape.")
    def test_fft_ops(self):
        x_val = make_xval([3, 4]).astype(np.float32)
        def func1(x):
            xc = tf.cast(x, tf.complex64)
            op_ = tf.signal.fft(xc)
            return tf.abs(op_, name=_TFOUTPUT)
        self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "topk")
    def test_invert_permutation(self):

        def func(x):
            op_ = tf.math.invert_permutation(x)
            return tf.identity(op_, name=_TFOUTPUT)

        x_val = np.array([0, 1, 2, 3], dtype=np.int64)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array([1, 5, 2, 0, 3, 4], dtype=np.int64)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_tf_min_version("1.14")
    @check_opset_min_version(11, "CumSum")
    def test_rfft2d_ops_specific_dimension(self):

        x_val = make_xval([3, 1, 4]).astype(np.float32)

        def func1(x):
            op_ = tf.signal.rfft2d(x, np.array([1, 4], dtype=np.int32))
            return tf.abs(op_, name=_TFOUTPUT)
        with self.subTest(shape=(3, 1, 4), fft_length=(1, 4), optimize=False):
            self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val}, optimize=False)
        with self.subTest(shape=(3, 1, 4), fft_length=(1, 4), optimize=True):
            self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val})

        for shape in [(3, 1, 4), (5, 7), (3, 5, 7), (7, 5)]:
            for fft_length in [shape[-2:], (1, shape[-1]),
                               (min(2, shape[-2]), shape[-1]),
                               (shape[-2], 2),
                               (min(3, shape[-2]), min(4, shape[-2]))]:
                if fft_length == (1, 1):
                    # The code fails in this case but that's unlikely to happen.
                    continue
                for optimize in [False, True]:
                    with self.subTest(shape=shape, fft_length=fft_length, optimize=optimize):
                        x_val = make_xval(list(shape)).astype(np.float32)
                        x_val /= x_val.size
                        def func1(x):
                            op_ = tf.signal.rfft2d(x, np.array(fft_length, dtype=np.int32))
                            return tf.abs(op_, name=_TFOUTPUT)
                        self._run_test_case(func1, [_OUTPUT], {_INPUT: x_val}, optimize=optimize)

    @check_tf_min_version("2.1")
    @skip_tflite("TFlite errors on some attributes")
    @check_opset_min_version(9, "string")
    def test_asstring(self):
        def func(x):
            op_ = tf.strings.as_string(x)
            return tf.identity(op_, name=_TFOUTPUT)

        x_val = np.array([0, 1, 2, 3], dtype=np.int32)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array([0, 1, 2, 3], dtype=np.float32)
        # can't check the values because in onnx they are padded with 0, in tf they are not
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_value=False)

    @check_tf_min_version("2.1")
    @skip_tflite("TFlite errors on some attributes")
    @check_opset_min_version(9, "string")
    def test_string_to_number(self):
        def func(x):
            op_ = tf.strings.to_number(x)
            return tf.identity(op_, name=_TFOUTPUT)

        # tf gets this wrong and returns fp32 instead of int
        x_val = np.array("123", dtype=object)
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

        x_val = np.array("123.1", dtype=object)
        # can't check the values because in onnx they are padded with 0, in tf they are not
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val}, check_value=False)

    @check_tf_min_version("2.5")
    @check_opset_min_version(14, "hardswish")
    @skip_tfjs("not supported in tfjs")
    def test_hardswish(self):
        def func(x):
            # there is no hardswich in tf but toco will optimize to it
            op_ = x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)
            return tf.identity(op_, name=_TFOUTPUT)

        # tf gets this wrong and returns fp32 instead of int
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @skip_tfjs("not supported in tfjs")
    def test_l2normalization(self):
        def func(x):
            op_ = tf.math.l2_normalize(x)
            return tf.identity(op_, name=_TFOUTPUT)

        x_val = make_xval([3, 4])
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_addition_two_newaxis_simultaneously(self):
        def func(x):
            op = x[..., tf.newaxis, tf.newaxis]
            return tf.identity(op, name=_TFOUTPUT)

        x_val = make_xval([2, 3])
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice")
    def test_addition_three_newaxis_simultaneously(self):
        def func(x):
            op = x[..., tf.newaxis, tf.newaxis, tf.newaxis]
            return tf.identity(op, name=_TFOUTPUT)

        x_val = make_xval([2, 3])
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(11, "Pad")
    def test_conv_unknown_kernel_channels(self):
        x_shape = [2, 10, 3]
        x_val = make_xval(x_shape)
        kernel_shape = [4, 3, 5]
        kernel_val = make_xval(kernel_shape)
        pad_val = np.array([[0, 0], [0, 0], [0, 0]], np.int64)
        def func(x, kernel, pad):
            # Make kernel dimensions unknown
            kernel = tf.pad(kernel, pad)
            conv = tf.nn.conv1d(x, kernel, stride=[1], padding='VALID')
            return tf.identity(conv, name='output')
        self._run_test_case(func, [_OUTPUT], {_INPUT: x_val, _INPUT1: kernel_val, _INPUT2: pad_val})

    @check_tf_min_version("2.3.0")
    @check_opset_min_version(16, "ScatterND")
    @skip_tfjs("not supported in tfjs")
    def test_tensor_scatter_max(self):
        def func(tensor, indices, updates):
            op = tf.tensor_scatter_nd_max(tensor, indices, updates)
            return tf.identity(op, name=_TFOUTPUT)

        tensor_val = make_xval([3, 4, 5])
        indices_val = np.array([[2, 3], [0, 1]], np.int32)
        indices64_val = indices_val.astype(np.int64)
        updates_val = make_xval([2, 5]) + 3
        self._run_test_case(func, [_OUTPUT], {_INPUT: tensor_val, _INPUT1: indices_val, _INPUT2: updates_val})
        self._run_test_case(func, [_OUTPUT], {_INPUT: tensor_val, _INPUT1: indices64_val, _INPUT2: updates_val})

    @check_tf_min_version("2.3.0")
    @check_opset_min_version(16, "ScatterND")
    @skip_tfjs("not supported in tfjs")
    def test_tensor_scatter_min(self):
        def func(tensor, indices, updates):
            op = tf.tensor_scatter_nd_min(tensor, indices, updates)
            return tf.identity(op, name=_TFOUTPUT)

        tensor_val = make_xval([3, 4, 5])
        indices_val = np.array([[2, 3], [0, 1]], np.int32)
        indices64_val = indices_val.astype(np.int64)
        updates_val = make_xval([2, 5]) + 3
        self._run_test_case(func, [_OUTPUT], {_INPUT: tensor_val, _INPUT1: indices_val, _INPUT2: updates_val})
        self._run_test_case(func, [_OUTPUT], {_INPUT: tensor_val, _INPUT1: indices64_val, _INPUT2: updates_val})

    @check_tf_min_version("1.12.1")
    @check_opset_min_version(16, "ScatterND")
    @skip_tfjs("not supported in tfjs")
    def test_tensor_scatter_sub(self):
        def func(tensor, indices, updates):
            op = tf.tensor_scatter_nd_sub(tensor, indices, updates)
            return tf.identity(op, name=_TFOUTPUT)

        tensor_val = make_xval([3, 4, 5])
        indices_val = np.array([[2, 3], [0, 1]], np.int32)
        indices64_val = indices_val.astype(np.int64)
        updates_val = make_xval([2, 5]) + 3
        self._run_test_case(func, [_OUTPUT], {_INPUT: tensor_val, _INPUT1: indices_val, _INPUT2: updates_val})
        self._run_test_case(func, [_OUTPUT], {_INPUT: tensor_val, _INPUT1: indices64_val, _INPUT2: updates_val})

if __name__ == '__main__':
    unittest_main()
