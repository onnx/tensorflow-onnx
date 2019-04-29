# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit tests using onnx backends."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from itertools import product

import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
# pylint reports unused-wildcard-import which is false positive, __all__ is defined in common
from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tf2onnx import constants

# pylint: disable=missing-docstring,invalid-name,unused-argument


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]

_STRIDE1x1 = [1, 1, 1, 1]
_KERNEL3x3 = [3, 3, 1, 1]

# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFINPUT2 = "input2"
_INPUT2 = "input2:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"
_TFOUTPUT1 = "output1"
_OUTPUT1 = "output1:0"


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


class BackendTests(Tf2OnnxBackendTestBase):
    def _run_test_case(self, output_names_with_port, feed_dict, **kwargs):
        kwargs["convert_var_to_const"] = False
        kwargs["constant_fold"] = False
        return self.run_test_case(feed_dict, [], output_names_with_port, **kwargs)

    def _test_expand_dims(self, idx):
        tf.reset_default_graph()
        x_val = make_xval([3, 4])
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.expand_dims(x, idx)
        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_expand_dims(self):
        for i in [-1, 0, 1, -2]:
            self._test_expand_dims(i)

    def test_expand_dims_dynamic_inputs(self):
        x_val = make_xval([3, 4])
        x = tf.placeholder(tf.float32, shape=[None, None], name=_TFINPUT)
        op = tf.expand_dims(x, 0)
        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_expand_dims_one_unknown_rank(self):
        tf.reset_default_graph()
        x_val = make_xval([3, 4])
        x = tf.placeholder(tf.float32, shape=[None, 4], name=_TFINPUT)
        op = tf.expand_dims(x, 0)
        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_expand_dims_more_unknown_rank(self):
        tf.reset_default_graph()
        x_val = make_xval([3, 4])
        x = tf.placeholder(tf.float32, shape=[None, None], name=_TFINPUT)
        op = tf.expand_dims(x, 0)
        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "trig")
    def test_trig_ops(self):
        for op in [tf.sin, tf.cos, tf.tan, tf.asin, tf.acos, tf.atan]:
            tf.reset_default_graph()
            x_val = make_xval([3, 4])
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            op_ = op(x)
            _ = tf.identity(op_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-06)

    @check_opset_min_version(9, "trigh")
    def test_atrig_ops(self):
        for op in [tf.sinh, tf.cosh, tf.atanh, tf.asinh, tf.acosh]:
            tf.reset_default_graph()
            x_val = make_xval([3, 4])
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            op_ = op(x)
            _ = tf.identity(op_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_opset_min_version(7, "multinomial")
    def test_multinomial(self):
        x_val = np.array([[10., 10.]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.multinomial(tf.log(x), 5, output_dtype=tf.int64)
        _ = tf.identity(op, name=_TFOUTPUT)

        # since returned indexes are random we can only check type and shape
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_value=False,
                            check_shape=True, check_dtype=True)

    @skip_caffe2_backend()
    @check_opset_min_version(7, "multinomial")
    def test_multinomial1(self):
        shape = [2, 10]
        x_val = np.ones(np.prod(shape)).astype("float32").reshape(shape)
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.multinomial(x, 2, output_dtype=tf.int64)
        _ = tf.identity(op, name=_TFOUTPUT)
        # since returned indexes are random we can only check type and shape
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_value=False,
                            check_shape=True, check_dtype=True)

    def test_maxpool(self):
        for tf_shape in ["known", "unknown"]:
            tf.reset_default_graph()
            for p in get_conv_getdata():
                _, padding, x_shape, ksize, strides = p
                tf.reset_default_graph()
                x_val = make_xval(x_shape)
                if tf_shape == "known":
                    x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
                else:
                    x = tf.placeholder(tf.float32, shape=[None] * x_val.ndim, name=_TFINPUT)
                mp = tf.nn.max_pool(x, ksize, strides, padding=padding)
                _ = tf.identity(mp, name=_TFOUTPUT)

                self.logger.debug(str(p))
                self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("AveragePool")
    def test_avgpool(self):
        for tf_shape in ["known", "unknown"]:
            for p in get_conv_getdata(kind=0):
                _, padding, x_shape, ksize, strides = p
                tf.reset_default_graph()
                x_val = make_xval(x_shape)
                if tf_shape == "known":
                    x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
                else:
                    x = tf.placeholder(tf.float32, shape=[None] * x_val.ndim, name=_TFINPUT)
                mp = tf.nn.avg_pool(x, ksize, strides, padding=padding)
                _ = tf.identity(mp, name=_TFOUTPUT)

                self.logger.debug(str(p))
                self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-06)

    def _conv_test(self, x_val, w, strides=None, padding="VALID", dilations=None, rtol=1e-07):
        if strides is None:
            strides = _STRIDE1x1
        if dilations is None:
            dilations = [1, 1, 1, 1]
        tf.reset_default_graph()
        kernel = tf.constant(w, dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.conv2d(x, kernel, strides=strides, padding=padding, dilations=dilations)
        _ = tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=rtol)

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
        self._conv_test(x_val, kernel_val, strides=strides, padding="VALID", rtol=1e-05)

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
            tf.reset_default_graph()
            x_val = make_xval((1, 1, *input_shape)).transpose(NCHW_TO_NHWC)
            w = np.random.random_sample([3, 3, 1, 2]).astype(np.float32)
            strides = [1, 2, 2, 1]

            x = tf.placeholder(tf.float32, shape=[None] * 4, name=_TFINPUT)
            kernel = tf.constant(w, dtype=tf.float32, name='k')
            conv = tf.nn.conv2d(x, kernel, strides=strides, padding="SAME")
            _ = tf.identity(conv, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-5)

    def test_conv2d_with_pad(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.random.random_sample([3, 3, 1, 2]).astype(np.float32)
        strides = [1, 1, 1, 1]

        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        kernel = tf.constant(w, dtype=tf.float32, name='k')
        x_pad = tf.pad(x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])
        conv = tf.nn.conv2d(x_pad, kernel, strides=strides, padding="VALID")
        _ = tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-5)

    def test_conv2d_transpose(self):
        x_shape = [2, 6, 4, 3]
        output_shape = [2, 13, 9, 2]
        kernel_shape = [3, 3, 2, 3]
        strides = [1, 2, 2, 1]
        x_val = make_xval(x_shape)
        kernel_val = make_xval(kernel_shape)
        x = tf.placeholder(tf.float32, shape=x_shape, name=_TFINPUT)
        f = tf.constant(kernel_val, name="kernel", dtype=tf.float32)
        conv = tf.nn.conv2d_transpose(x, f, output_shape, strides=strides, padding="VALID")
        _ = tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    def test_depthwiseconv_0(self):
        x_shape = [1, 3, 4, 3]
        kernel_shape = [3, 3, 3, 3]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        _ = tf.identity(conv, name=_TFOUTPUT)
        # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=0.08)

    def test_depthwiseconv_1(self):
        x_shape = [1, 112, 112, 32]
        kernel_shape = [3, 3, 32, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.depthwise_conv2d(x, kernel, strides=_STRIDE1x1, padding='VALID')
        _ = tf.identity(conv, name=_TFOUTPUT)
        # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=0.08)

    def test_dropout(self):
        is_training = tf.placeholder_with_default(False, (), "is_training")
        x_val = np.ones([1, 24, 24, 3], dtype=np.float32)
        # Define a scope for reusing the variables
        x = tf.placeholder(tf.float32, shape=x_val.shape, name="input_1")
        x_ = tf.identity(x)

        fc1 = tf.layers.dropout(x_, rate=.1, training=is_training)

        _ = tf.identity(fc1, name="output")
        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_nn_dropout(self):
        keep_prob = tf.placeholder_with_default(1., (), "keep_prob")
        x_val = np.ones([1, 24, 24, 3], dtype=np.float32)
        # Define a scope for reusing the variables
        x = tf.placeholder(tf.float32, shape=x_val.shape, name="input_1")
        x_ = tf.identity(x)

        fc1 = tf.nn.dropout(x_, keep_prob)

        _ = tf.identity(fc1, name="output")
        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        # when constant_fold is enabled, PlaceholderWithDefault will be folded into either a const or a placeholder.
        # here we set it False to test PlaceholderWithDefault bug: https://github.com/onnx/tensorflow-onnx/pull/446
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, constant_fold=False)

    @check_tf_min_version("1.13")
    def test_nn_dropout_with_rate(self):
        rate = tf.placeholder_with_default(0., (), "rate")
        x_val = np.ones([1, 24, 24, 3], dtype=np.float32)
        # Define a scope for reusing the variables
        x = tf.placeholder(tf.float32, shape=x_val.shape, name="input_1")
        x_ = tf.identity(x)

        fc1 = tf.nn.dropout(x_, rate=rate)

        _ = tf.identity(fc1, name="output")
        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, constant_fold=False)

    def test_conv2d_with_input_transpose(self):
        x_shape = [2, 32, 32, 3]
        kernel_shape = [3, 3, 3, 3]
        x_val = make_xval(x_shape)
        x_val_for_onnx = x_val.transpose(NHWC_TO_NCHW)
        kernel = tf.constant(make_xval(kernel_shape), dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
        _ = tf.identity(conv, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05,
                            process_args={"inputs_as_nchw": [_INPUT]},
                            onnx_feed_dict={_INPUT: x_val_for_onnx})

    @unittest.skip("")
    def test_lrn(self):
        # FIXME: numerical results are not correct
        x_shape = [1, 3, 4, 3]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        _ = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.nn.local_response_normalization(x_val)
        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_incompatibility("Abs")
    def test_abs(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.abs(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Add")
    def test_const(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        y = tf.constant(x_val, name="y")
        _ = tf.add(x, y, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Add")
    def test_add(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.add(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_placeholder(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_placeholder_with_default_use_default(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.constant(x_val, name="x")
        y = tf.placeholder_with_default(x, x_val.shape, name=_TFINPUT)
        _ = tf.identity(y, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})

    def test_placeholder_with_default_use_feed(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.constant(x_val, name="x")
        y = tf.placeholder_with_default(x, x_val.shape, name=_TFINPUT)
        _ = tf.identity(y, name=_TFOUTPUT)
        x_feed_val = np.array([11.0, 22.0, -33.0, -44.0], dtype=np.float32).reshape((2, 2))
        self._run_test_case([_OUTPUT], {_INPUT: x_feed_val})

    @check_onnxruntime_incompatibility("Add")
    def test_add_bcast(self):
        x1_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x2_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32).reshape((2, 2, 2))
        # if we'd broadcast 2,2 to 2,1 onnxmsrt will fail
        x1 = tf.placeholder(tf.float32, x1_val.shape, name="input")
        x2 = tf.placeholder(tf.float32, x2_val.shape, name=_TFINPUT1)
        x_ = tf.add(x1, x2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x1_val, _INPUT1: x2_val})

    @check_onnxruntime_incompatibility("Add")
    def test_add_bcast1(self):
        # example taken from onnx doc
        x1_val = np.random.randn(3, 4, 5).astype(np.float32)
        x2_val = np.random.randn(5).astype(np.float32)
        x1 = tf.placeholder(tf.float32, x1_val.shape, name="input")
        x2 = tf.placeholder(tf.float32, x2_val.shape, name=_TFINPUT1)
        x_ = tf.add(x1, x2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x1_val, _INPUT1: x2_val})

    def test_matmul0(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.matmul(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_matmul1(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.matmul(x, x, transpose_a=True)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_matmul2(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        y_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        y = tf.placeholder(tf.float32, y_val.shape, name=_TFINPUT1)
        x_ = tf.matmul(x, y, transpose_b=True)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @unittest.skipIf(get_test_config().is_mac and get_test_config().is_onnxruntime_backend
                     and get_test_config().backend_version == "0.2.1", "onnxruntime 0.2.1 has bug on mac")
    def test_matmul3(self):
        x_shape = [1, 12, 256, 64]
        x_val = np.arange(np.prod(x_shape)).astype("float32").reshape((x_shape))
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        y = tf.placeholder(tf.float32, x_shape, name=_TFINPUT1)
        x_ = tf.matmul(x, y, transpose_b=True)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: x_val}, rtol=1e-5)

    @check_onnxruntime_incompatibility("Sub")
    def test_sub(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.subtract(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Mul")
    def test_multiply(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.multiply(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Div")
    def test_div(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.realdiv(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Exp")
    def test_exp(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.exp(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_incompatibility("Log")
    def test_log(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.log(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_gather(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        idx = np.array([1, 0, 2], dtype=np.int32)
        idx_flattened = np.array([i * x_val.shape[1] + idx for i in range(0, x_val.shape[0])])
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather(tf.reshape(x, [-1]), tf.constant(idx_flattened))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_target('rs6', 'GatherNd')
    def test_gathernd(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        indices = np.array([[[0, 1], [1, 1]], [[1, 2], [0, 2]]], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather_nd(x, tf.constant(indices))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        indices = np.array([[[0], [2]], [[4], [7]], [[6], [1]]], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather_nd(x, tf.constant(indices))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_target('rs6', 'GatherNd')
    def test_gathernd_less_index(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        indices = np.array([[[0], [1]], [[2], [0]]], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather_nd(x, tf.constant(indices))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        # shape: 2*2*2
        x_val = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        indices = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather_nd(x, tf.constant(indices))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_opset_min_version(7, "tile")
    def test_tile(self):
        x_val = np.array([[0, 1], [2, 3]], dtype=np.float32)
        multiple = tf.constant([2, 2])
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.tile(x, multiple)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Neg")
    def test_neg(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.negative(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Mul")
    def test_square(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.square(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Min")
    def test_min(self):
        x_val1 = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32).reshape((2, 2))
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, x_val2.shape, name=_TFINPUT1)
        mi = tf.minimum(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

        tf.reset_default_graph()
        x_val1 = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.int32).reshape((2, 2))
        x1 = tf.placeholder(tf.int32, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.int32, x_val2.shape, name=_TFINPUT1)
        mi = tf.minimum(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @skip_caffe2_backend("issue with broadcasting scalar")
    @check_onnxruntime_incompatibility("Sub")
    def test_min_broadcast(self):
        # tests if the broadcast for min/max is working
        x_val1 = np.array([2.0, 16.0, 5.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.constant(x_val2, dtype=tf.float32, name='x2')
        mi = tf.minimum(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1})

    @check_onnxruntime_incompatibility("Add")
    def test_logicaland(self):
        x_val1 = np.array([1, 0, 1, 1], dtype=np.bool).reshape((2, 2))
        x_val2 = np.array([0, 1, 1, 1], dtype=np.bool).reshape((2, 2))
        x1 = tf.placeholder(tf.bool, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.bool, [2, 2], name=_TFINPUT1)
        mi = tf.logical_and(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Greater")
    def test_greater(self):
        for op in [tf.greater, tf.greater_equal]:
            tf.reset_default_graph()
            x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
            x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
            x1 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
            x2 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT1)
            mi = op(x1, x2)
            _ = tf.identity(mi, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Greater")
    def test_greater_unsupport_type(self):
        for op in [tf.greater, tf.greater_equal]:
            tf.reset_default_graph()
            x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
            x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
            x1 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
            x2 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT1)
            mi = op(x1, x2)
            _ = tf.identity(mi, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Less")
    def test_less(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
        x1 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT1)
        mi = tf.less(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Less")
    def test_less_unsupport_type(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        x1 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT1)
        mi = tf.less(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @check_onnxruntime_incompatibility("Equal")
    def test_equal(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        x1 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT1)
        mi = tf.equal(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

        tf.reset_default_graph()
        x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
        x1 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT1)
        mi = tf.equal(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    def test_sequeeze_no_axis_specified(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2, 1))
        x = tf.placeholder(tf.float32, [2, 2, 1], name=_TFINPUT)
        x_ = tf.squeeze(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_sequeeze_positive_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2, 1))
        x = tf.placeholder(tf.float32, [2, 2, 1], name=_TFINPUT)
        x_ = tf.squeeze(x, [2])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_sequeeze_negative_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2, 1))
        x = tf.placeholder(tf.float32, [2, 2, 1], name=_TFINPUT)
        x_ = tf.squeeze(x, [-1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_sequeeze_mixed_axis(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((1, 2, 2, 1))
        x = tf.placeholder(tf.float32, [1, 2, 2, 1], name=_TFINPUT)
        x_ = tf.squeeze(x, [0, -1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_transpose(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape((2, 3))
        x = tf.placeholder(tf.float32, [2, 3], name=_TFINPUT)
        x_ = tf.transpose(x)  # perm=[1,0])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_reshape(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        shape = tf.constant([1, 4])
        x_ = tf.reshape(x, shape)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_shape=True)

    @check_opset_min_version(6, "cast")
    def test_reshape_int(self):
        x_val = np.array([1, 2, 3, 4], dtype=np.int32).reshape((2, 2))
        x = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
        shape = tf.constant([1, 4])
        x_ = tf.reshape(x, shape)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_shape=True)

    @unittest.skipIf(get_test_config().opset < 5 or get_test_config().backend in ["onnxmsrtnext"],
                     "since opset 5, broken in msrtnext")
    @check_opset_min_version(6, "cast")
    def test_reshape_dynamic(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        shape_val = np.array([4, 1], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        shape = tf.placeholder(tf.int32, shape_val.shape, name=_TFINPUT1)
        x_ = tf.reshape(x, shape)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: shape_val}, check_shape=True)

    @check_onnxruntime_incompatibility("Relu")
    def test_relu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.relu(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("fails on caffe2 with dim issue")
    @check_onnxruntime_incompatibility("Mul")
    @check_tf_min_version("1.6")
    def test_leaky_relu_int(self):
        # starting from tf 1.6, leaky_relu supports `feature` x of int type
        x_types = [np.int32, np.int64]
        for x_type in x_types:
            x_val = 1000 * np.random.random_sample([1000, 100]).astype(x_type)
            for alpha in [0.1, -0.1, 1.0, -1.0]:
                x = tf.placeholder(x_val.dtype, [None] * x_val.ndim, name=_TFINPUT)
                x_ = tf.nn.leaky_relu(x, alpha)
                _ = tf.identity(x_, name=_TFOUTPUT)
                self._run_test_case([_OUTPUT], {_INPUT: x_val})
                tf.reset_default_graph()

    @skip_caffe2_backend("fails on caffe2 with dim issue")
    @check_onnxruntime_incompatibility("Mul")
    def test_leaky_relu_float(self):
        x_val = 1000 * np.random.random_sample([1000, 100]).astype(np.float32)
        for alpha in [0.1, -0.1, 1.0, -1.0]:
            x = tf.placeholder(x_val.dtype, [None] * x_val.ndim, name=_TFINPUT)
            x_ = tf.nn.leaky_relu(x, alpha)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})
            tf.reset_default_graph()

    @check_onnxruntime_incompatibility("Elu")
    def test_elu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.elu(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Tanh")
    def test_tanh(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.tanh(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_incompatibility("Max")
    def test_relu6(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.relu6(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Sub")
    def test_relu6_dynamic(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [None, 2], name=_TFINPUT)
        x_ = tf.nn.relu6(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_concat(self):
        x_val1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x_val2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        x_val3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, x_val2.shape, name=_TFINPUT1)
        x3 = tf.placeholder(tf.float32, x_val3.shape, name="input3")
        x_ = tf.concat([x1, x2, x3], 0)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, "input3:0": x_val3})

    def test_concat_empty_const_input(self):
        x_val1 = np.array([1, 2, 3], dtype=np.float32)
        x_val2 = np.array([], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.constant(x_val2, dtype=tf.float32)
        x_ = tf.concat([x1, x2], 0)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1})

        tf.reset_default_graph()
        x_val1 = np.array([[1, 2, 3]], dtype=np.float32)
        x_val2 = np.array([[]], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.constant(x_val2, dtype=tf.float32)
        x_ = tf.concat([x1, x2], 1)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1})

        tf.reset_default_graph()
        x_val1 = np.array([1, 2, 3], dtype=np.float32)
        x_val2 = np.array([], dtype=np.float32)
        x_val3 = np.array([13, 14, 15], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.constant(x_val2, dtype=tf.float32)
        x3 = tf.placeholder(tf.float32, x_val3.shape, name=_TFINPUT1)
        x_ = tf.concat([x1, x2, x3], 0)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val3})

    @check_opset_min_version(6, "cast")
    def test_concat_int64(self):
        x_val1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        x_val2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64)
        x_val3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.int64)
        x1 = tf.placeholder(tf.int64, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.int64, x_val2.shape, name=_TFINPUT1)
        x3 = tf.placeholder(tf.int64, x_val3.shape, name="input3")
        x_ = tf.concat([x1, x2, x3], 0)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, "input3:0": x_val3})

    def test_concat_negative_axis(self):
        x_val1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x_val2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        x_val3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, x_val2.shape, name=_TFINPUT1)
        x3 = tf.placeholder(tf.float32, x_val3.shape, name="input3")
        x_ = tf.concat([x1, x2, x3], -1)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2, "input3:0": x_val3})

    @check_onnxruntime_incompatibility("Pow")
    def test_pow(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
        e = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.pow(x, tf.constant(e))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_embedding_lookup(self):
        x_val1 = np.array([[1]], dtype=np.int32)
        x_val2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        t = tf.constant(x_val2)
        x = tf.placeholder(tf.int32, x_val1.shape, name=_TFINPUT)
        x_ = tf.nn.embedding_lookup(t, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1})

    def test_slice(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        t1 = tf.constant([0, 1], dtype=tf.int32)
        t2 = tf.constant([2, 2], dtype=tf.int32)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.slice(x0, t1, t2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "Slice in opset 10 can accept dymaic 'start' and 'ends'")
    def test_slice_with_non_const(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        t1 = np.array([0, 1], dtype=np.int32)
        t2 = np.array([2, 2], dtype=np.int32)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        t1_ = tf.placeholder(tf.int32, t1.shape, name=_TFINPUT1)
        t2_ = tf.placeholder(tf.int32, t2.shape, name=_TFINPUT2)
        x_ = tf.slice(x0, t1_, t2_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: t1, _INPUT2: t2})

    @check_opset_min_version(10, "Slice in opset 10 can accept dymaic 'start' and 'ends'")
    def test_slice_with_size_is_negative_one(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        t1 = np.array([0, 1], dtype=np.int32)
        # input "size" contains -1
        t2 = np.array([2, -1], dtype=np.int32)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        t1_ = tf.placeholder(tf.int32, t1.shape, name=_TFINPUT1)
        t2_ = tf.placeholder(tf.int32, t2.shape, name=_TFINPUT2)
        x_ = tf.slice(x0, t1_, t2_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: t1, _INPUT2: t2})

    @skip_caffe2_backend()
    def test_slice1(self):
        # FIXME: only 1 dimension supported by caffe2 and msrt
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
        t1 = tf.constant([1, 0, 0], dtype=tf.int32)
        t2 = tf.constant([1, 1, 3], dtype=tf.int32)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.slice(x0, t1, t2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_split(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape((5, 30))
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_, _, _ = tf.split(x0, [4, 15, 11], 1)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_split_with_more_outputs(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape((5, 30))
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        _, _, _ = tf.split(x0, [4, 15, 11], 1, name="split_test")
        self._run_test_case(["split_test:0", "split_test:1", "split_test:2"], {_INPUT: x_val})

    def test_reducesum(self):
        # not supported by onnx-caffe2
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reduce_sum(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_onnxruntime_incompatibility("Sqrt")
    def test_sqrt(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.sqrt(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def _test_range_const(self, extra_opset=None):
        process_args = {}
        if extra_opset is not None:
            process_args["extra_opset"] = [extra_opset]

        x = tf.range(5)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {}, process_args=process_args)
        tf.reset_default_graph()

        x = tf.range(3, 3, 5)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {}, process_args=process_args)
        tf.reset_default_graph()

        x = tf.range(0, -5, -2)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {}, process_args=process_args)
        tf.reset_default_graph()

        x = tf.range(-5.0, 5.0, 1.5)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {}, process_args=process_args)
        tf.reset_default_graph()

        x = tf.range(2.5, 5.0, 10.0)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {}, process_args=process_args)

    def _test_range_non_const(self, extra_opset=None):
        process_args = {}
        if extra_opset is not None:
            process_args["extra_opset"] = [extra_opset]

        x = tf.range(5.0)
        _ = tf.identity(x, name=_TFOUTPUT)
        g = self._run_test_case([_OUTPUT], {}, process_args=process_args)
        self.assertTrue(extra_opset is None
                        or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))
        tf.reset_default_graph()

        x = tf.range(0, -5.0, -2)
        _ = tf.identity(x, name=_TFOUTPUT)
        g = self._run_test_case([_OUTPUT], {}, process_args=process_args)
        self.assertTrue(extra_opset is None
                        or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))
        tf.reset_default_graph()

        # disable this case for ms domain due to onnxruntime range-1 issue
        # https://github.com/Microsoft/onnxruntime/issues/730
        if not (extra_opset and extra_opset.domain == constants.MICROSOFT_DOMAIN):
            x = tf.range(3.0, 3.0, 5)
            _ = tf.identity(x, name=_TFOUTPUT)
            g = self._run_test_case([_OUTPUT], {}, process_args=process_args)
            self.assertTrue(extra_opset is None
                            or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))
            tf.reset_default_graph()

        delta_val = np.array(1.5, dtype=np.float32)
        delta = tf.placeholder(tf.float32, shape=(), name=_TFINPUT)
        x = tf.range(-5.0, 5.0, delta)
        _ = tf.identity(x, name=_TFOUTPUT)
        g = self._run_test_case([_OUTPUT], {_INPUT: delta_val}, process_args=process_args)
        self.assertTrue(extra_opset is None
                        or check_node_domain(group_nodes_by_type(g)["Range"][0], extra_opset.domain))
        tf.reset_default_graph()

        start_val = np.array(2.5, dtype=np.float32)
        start = tf.placeholder(tf.float32, shape=(), name=_TFINPUT)
        x = tf.range(start, 5.0, 10.0)
        _ = tf.identity(x, name=_TFOUTPUT)
        g = self._run_test_case([_OUTPUT], {_INPUT: start_val}, process_args=process_args)
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
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.rsqrt(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @check_onnxruntime_incompatibility("Reciprocal")
    def test_reciprocal(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reciprocal(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-04)

    def test_reducemax(self):
        # not supported by onnx-caffe2
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.reduce_max(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @skip_caffe2_backend()
    def test_reduceprod(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reduce_prod(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_reducemean(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reduce_mean(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    @check_onnxruntime_incompatibility("Pow")
    def test_pow_scalar(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
        e = np.array(2.0, dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.pow(x, tf.constant(e))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_pad_const_default_val(self):
        params = [
            ("CONSTANT", [[1, 1], [2, 2]], [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]),
            ("CONSTANT", [[0, 0], [3, 3], [3, 3], [0, 0]], np.random.randn(1, 3, 4, 5).astype(np.float32)),
        ]
        for p in params:
            tf.reset_default_graph()
            mode, pad, xv = p
            x_val = np.array(xv, dtype=np.float32)
            x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
            paddings = tf.constant(pad)
            op = tf.pad(x, paddings, mode)
            _ = tf.identity(op, name=_TFOUTPUT)
            self.logger.debug(str(p))
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_pad_const(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        paddings = tf.constant([[1, 1], [2, 2]], name="paddings")
        op = tf.pad(x, paddings, mode="CONSTANT", name="const_with_val", constant_values=999)

        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_pad_reflect(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        paddings = tf.constant([[1, 1], [2, 2]], name="paddings")
        op = tf.pad(x, paddings, mode="REFLECT", name="reflect")

        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend()
    def test_randomuniform(self):
        shape = tf.constant([2, 3], name="shape")
        x_ = tf.random_uniform(shape, name="rand", dtype=tf.float32)
        x_ = tf.identity(x_, name="output1")
        x_ = tf.identity(x_, name="output2")
        _ = tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case([_OUTPUT], {}, check_value=False, check_shape=True)

    @unittest.skip("TF RandomUniformInt is not supported")
    def test_randomuniform_int(self):
        shape = tf.constant([2, 3], name="shape")
        x_ = tf.random_uniform(shape, name="rand", dtype=tf.int32, maxval=10)
        x_ = tf.identity(x_, name="output1")
        x_ = tf.identity(x_, name="output2")
        _ = tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case([_OUTPUT], {}, check_value=False, check_shape=True)

    @skip_caffe2_backend()
    def test_randomuniform_dyn_shape(self):
        # test for dynamic shape coming from a shape op
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        x = tf.placeholder(x_val.dtype, name=_TFINPUT)
        x_ = tf.stack([x, x])
        x_ = tf.identity(x_)
        x_ = tf.shape(x_, name="shape")
        x_ = tf.random_uniform(x_, name="rand", dtype=tf.float32)
        x_ = tf.identity(x_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_value=False, check_shape=True)

    @skip_caffe2_backend()
    def test_randomuniform_calc_shape(self):
        # test for dynamic shape coming from some subgraph
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        x = tf.placeholder(x_val.dtype, [None, 3], name=_TFINPUT)
        x_ = tf.identity(x)
        x_ = tf.shape(x_, name="shape")[1:]
        x_ = tf.random_uniform(x_, name="rand", dtype=tf.float32)
        x_ = tf.identity(x_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_value=False, check_shape=True)

    @skip_caffe2_backend()
    def test_argminmax(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(x_val.dtype, x_val.shape, name=_TFINPUT)
        x_ = tf.argmin(x, axis=0)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val = np.array([1, 2, -2, -1], dtype=np.int32).reshape((2, 2))
        x = tf.placeholder(x_val.dtype, x_val.shape, name=_TFINPUT)
        x_ = tf.argmax(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val = np.array([1, 2, -2, -1], dtype=np.int32).reshape((2, 2))
        x = tf.placeholder(x_val.dtype, x_val.shape, name=_TFINPUT)
        x_ = tf.argmax(x, output_type=x_val.dtype)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(6, "cast")
    def test_cast(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.cast(x, tf.int32)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_sign(self):
        x_val1 = np.array([1, 2, 0, -1, 0, -2], dtype=np.int32).reshape((2, 3))
        x_val2 = np.array([1, 2, 0, -1, 0, -2], dtype=np.int64).reshape((2, 3))
        x_val3 = np.array([1.0, 2.0, 0.0, -1.0, 0.0, -2.0], dtype=np.float32).reshape((2, 3))
        for x_val in [x_val1, x_val2, x_val3]:
            x = tf.placeholder(x_val.dtype, x_val.shape, name=_TFINPUT)
            x_ = tf.sign(x)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})
            tf.reset_default_graph()

    @check_target("rs6", "onehot")
    def test_onehot0(self):
        x_val = np.array([0, 1, 2], dtype=np.int32)
        depth = 5
        for axis in [-1, 0, 1]:
            tf.reset_default_graph()
            x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
            x_ = tf.one_hot(x, depth, on_value=5.0, axis=axis, off_value=1.0, dtype=tf.float32)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skip("only rank 1 is currently implemented")
    def test_onehot1(self):
        # only rank 1 is currently implemented
        x_val = np.array([[0, 2], [1, -1]], dtype=np.int32)
        depth = 3
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x_ = tf.one_hot(x, depth, on_value=5.0, axis=-1, off_value=0.0, dtype=tf.float32)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_target("rs6", "onehot")
    def test_onehot2(self):
        for axis in [-1, 0, 1]:
            tf.reset_default_graph()
            x_val = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 2], dtype=np.int32)
            depth = 20
            x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
            x_ = tf.one_hot(x, depth, on_value=5.0, axis=axis, off_value=1.0, dtype=tf.float32)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_target("rs6", "onehot")
    @check_opset_min_version(9, "onehot")
    def test_onehot3(self):
        # rank 1
        for np_dtype, tf_dtype in zip([np.int32, np.int64], [tf.int32, tf.int64]):
            tf.reset_default_graph()
            x_val = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 2], dtype=np_dtype)
            depth = np.array(20).astype(np.int64)
            x = tf.placeholder(tf_dtype, x_val.shape, name=_TFINPUT)
            on_off = np.array([5.6, 1.2]).astype(np_dtype)
            x_ = tf.one_hot(x, depth, on_value=on_off[0], axis=-1, off_value=on_off[1])
            _ = tf.identity(x_, name=_TFOUTPUT)
            graph = self._run_test_case([_OUTPUT], {_INPUT: x_val})
            self.assertTrue(len(group_nodes_by_type(graph)["OneHot"]) == 1, "onnx onehot should be used")
        # rank 2
        for aixs in [-1, 0, 1, 2]:
            for np_dtype, tf_dtype in zip([np.int32, np.int64], [tf.int32, tf.int64]):
                tf.reset_default_graph()
                x_val = np.arange(0, 50, dtype=np_dtype).reshape([-1, 10])
                depth = np.array(20).astype(np.int64)
                x = tf.placeholder(tf_dtype, x_val.shape, name=_TFINPUT)
                on_off = np.array([5.6, 1.2]).astype(np_dtype)
                x_ = tf.one_hot(x, depth, on_value=on_off[0], axis=aixs, off_value=on_off[1])
                _ = tf.identity(x_, name=_TFOUTPUT)
                graph = self._run_test_case([_OUTPUT], {_INPUT: x_val})
                self.assertTrue(len(group_nodes_by_type(graph)["OneHot"]) == 1, "onnx onehot should be used")

    @skip_caffe2_backend("issue undefined dim 1")
    def test_flatten0(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, [None, 3, 3], name=_TFINPUT)
        x_ = tf.layers.flatten(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_flatten1(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, [1, 3, 3], name=_TFINPUT)
        x_ = tf.layers.flatten(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_cancel_transpose(self):
        x_val = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.identity(x, _TFINPUT)
        x_ = tf.transpose(x_, perm=NHWC_TO_NCHW)
        x_ = tf.transpose(x_, perm=NCHW_TO_NHWC)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(6, "cast")
    def test_topk1(self):
        x_val = np.arange(3 * 2 * 3).astype("float32")
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        values, _ = tf.nn.top_k(x, 5, sorted=True)
        _ = tf.identity(values, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(10, "TopK with dynamic K")
    def test_topk2(self):
        x_val = np.arange(3 * 2 * 3).astype("float32")
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        k_val = np.array(10).astype(np.int32)
        k = tf.placeholder(tf.int32, name=_TFINPUT1)
        values, _ = tf.nn.top_k(x, k, sorted=True)
        _ = tf.identity(values, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: k_val})

    def test_stack_axis(self):
        for axis in [0, 1]:
            tf.reset_default_graph()
            x_val = [np.random.randn(3, 4).astype("float32") for _ in range(10)]
            x = [tf.constant(x_val[i], dtype=tf.float32) for i in range(10)]
            x_ = tf.stack(x, axis=axis)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {})

    def test_unstack_axis(self):
        for axis in [0, 1]:
            tf.reset_default_graph()
            x_val = np.random.randn(10, 3, 4).astype("float32")
            x = tf.constant(x_val, dtype=tf.float32)
            x_ = tf.unstack(x, axis=axis)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {})

    def _test_reorganize_data(self, op, shape):
        x_val = make_xval(shape)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = op(x, block_size=2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("Space2Depth not implemented")
    def test_space_to_depth(self):
        self._test_reorganize_data(tf.space_to_depth, [1, 2, 2, 1])

    @skip_caffe2_backend("Space2Depth not implemented")
    def test_depth_to_space(self):
        self._test_reorganize_data(tf.depth_to_space, [1, 1, 1, 4])

    @check_opset_min_version(6, "addn")
    def test_addn(self):
        x_val = np.arange(3 * 2 * 3).astype("float32")
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.add_n([x, x, x])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice1(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_strided_slice2(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [1, 0, 0], [2, 2, 3], [1, 1, 1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_strided_slice3(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[1:]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_strided_slice4(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[:2]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice5(self):
        x_val = np.arange(3 * 2 * 3).astype("float32").reshape((3, 2, 3))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[:2, 0:1, 1:]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice6(self):
        # example from here:
        # https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/ops/strided-slice
        x_val = np.arange(5 * 6).astype("float32").reshape((5, 6))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[2, :]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("multiple dims not supported")
    def test_strided_slice7(self):
        x_val = np.arange(5 * 6).astype("float32").reshape((5, 6))

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], begin_mask=2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], end_mask=2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], shrink_axis_mask=2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [0, 1], [3, 4], [1, 1], ellipsis_mask=2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @skip_caffe2_backend("fails with schema error")
    @check_opset_min_version(7, "batchnorm")
    def test_batchnorm(self):
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

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        scale = tf.constant(scale_val, name='scale')
        offset = tf.constant(offset_val, name='offset')
        mean = tf.constant(mean_val, name='mean')
        var = tf.constant(var_val, name='variance')
        epsilon = 0.001
        y, _, _ = tf.nn.fused_batch_norm(
            x, scale, offset, mean=mean, variance=var,
            epsilon=epsilon, data_format=data_format, is_training=False)
        _ = tf.identity(y, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-04)

    @skip_caffe2_backend()
    @check_opset_min_version(7, "resize_nearest_neighbor")
    def test_resize_nearest_neighbor(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        x_new_size_ = tf.constant(x_new_size)
        x_ = tf.image.resize_nearest_neighbor(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        graph = self._run_test_case([_OUTPUT], {_INPUT: x_val})
        if self.config.opset >= 9:
            # in opset 10, upsample is removed and resize is defined.
            node_statistic = group_nodes_by_type(graph)
            mapped_node = (node_statistic.get("Upsample") or node_statistic.get("Resize"))[0]
            scale_node = mapped_node.inputs[1]
            self.assertTrue(validate_const_node(scale_node, [1.0, 1.0, 2.0, 2.0]))

    @check_opset_min_version(9, "resize_nearest_neighbor")
    def test_resize_nearest_neighbor_with_non_const(self):
        x_shape = [3, 10, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape), dtype=np.float32).reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)

        x_new_size = np.array([20, 16]).astype(np.int32)
        x_new_size_ = tf.placeholder(shape=[None], dtype=tf.int32, name=_TFINPUT1)

        x_ = tf.image.resize_nearest_neighbor(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @skip_caffe2_backend()
    @check_opset_min_version(7, "resize_bilinear")
    def test_resize_bilinear(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        x_new_size_ = tf.constant(x_new_size)
        x_ = tf.image.resize_bilinear(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        graph = self._run_test_case([_OUTPUT], {_INPUT: x_val})
        if self.config.opset >= 9:
            # in opset 10, upsample is removed and resize is defined.
            node_statistic = group_nodes_by_type(graph)
            mapped_node = (node_statistic.get("Upsample") or node_statistic.get("Resize"))[0]
            scale_node = mapped_node.inputs[1]
            self.assertTrue(validate_const_node(scale_node, [1.0, 1.0, 2.0, 2.0]))

    @check_opset_min_version(9, "resize_bilinear")
    def test_resize_bilinear_with_non_const(self):
        x_shape = [3, 10, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape), dtype=np.float32).reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)

        x_new_size = np.array([20, 16]).astype(np.int32)
        x_new_size_ = tf.placeholder(shape=[None], dtype=tf.int32, name=_TFINPUT1)

        x_ = tf.image.resize_bilinear(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @check_opset_min_version(10, "resize scale can less than 1")
    def test_resize_bilinear_with_non_const2(self):
        # scales has an element larger than 1 and also has an element less that 1
        x_shape = [3, 100, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape), dtype=np.float32).reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)

        x_new_size = np.array([20, 16]).astype(np.int32)
        x_new_size_ = tf.placeholder(shape=[None], dtype=tf.int32, name=_TFINPUT1)

        x_ = tf.image.resize_bilinear(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @check_opset_min_version(10, "resize scale can less than 1")
    def test_resize_nearest_neighbor2(self):
        x_shape = [1, 300, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        x_new_size_ = tf.constant(x_new_size)
        x_ = tf.image.resize_nearest_neighbor(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        graph = self._run_test_case([_OUTPUT], {_INPUT: x_val})
        node_statistic = group_nodes_by_type(graph)
        mapped_node = node_statistic.get("Resize")[0]
        scale_node = mapped_node.inputs[1]
        self.assertTrue(validate_const_node(scale_node, [1.0, 1.0, 0.1, 2.0]))

    @check_opset_min_version(9, "fill")
    def test_fill_float32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9.0)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "fill")
    def test_fill_int32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("int32").reshape(x_shape)
        x0 = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "fill")
    def test_fill7_float32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9.0)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "fill")
    def test_fill7_int32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("int32").reshape(x_shape)
        x0 = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(7, "div")
    def test_tf_div(self):
        # pylint: disable=E0001
        from tensorflow.python.ops.gen_math_ops import div
        shape = 1000
        # test floating data
        x_val = (np.random.sample(shape) + 1e-6).astype(np.float32)
        y_val = (np.random.sample(shape) + 1e-6).astype(np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        y = tf.placeholder(tf.float32, y_val.shape, name=_TFINPUT1)
        output = div(x, y, name=_TFOUTPUT)
        assert output.op.type == "Div"
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

        tf.reset_default_graph()
        # test integer data
        x_val = (100 * np.random.sample(shape) + 1).astype(np.int32)
        y_val = (100 * np.random.sample(shape) + 1).astype(np.int32)
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        y = tf.placeholder(tf.int32, y_val.shape, name=_TFINPUT1)
        output = div(x, y, name=_TFOUTPUT)
        assert output.op.type == "Div"
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @check_opset_min_version(7, "erf")
    def test_erf(self):
        x_shape = [2, 2]
        x_val0 = np.random.random(np.prod(x_shape)).astype(np.float32).reshape(x_shape)
        x_val1 = np.array([[-1, -0.5], [1, 0.5]]).astype(np.float32)
        for x_val in [x_val0, x_val1]:
            tf.reset_default_graph()
            x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
            x_ = tf.erf(x)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=0.01)

    @check_opset_min_version(8, "Scan")
    @skip_opset(9, "ReverseSequence")
    def test_reverse_sequence_batch_major(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [0, 0, 0], [0, 0, 0]]],
                         dtype=np.float32)
        x = tf.placeholder(tf.float32, [None, 3, 3], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[2, 3, 1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3],
                          [4, 5, 6], [4, 5, 6], [1, 1, 1],
                          [0, 0, 0], [7, 8, 9], [0, 0, 0]
                          ],
                         dtype=np.float32)
        x = tf.placeholder(tf.float32, [None, 3], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[3] * 9)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val_shape = [5, 5, 7, 8, 9]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        x = tf.placeholder(tf.float32, [None, 5, 7, 8, 9], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[5, 5, 5, 5, 5])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "Scan")
    @skip_opset(9, "ReverseSequence")
    def test_reverse_sequence_time_major(self):
        x_val = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                          [[4, 5, 6], [4, 5, 6], [0, 0, 0]],
                          [[0, 0, 0], [7, 8, 9], [0, 0, 0]]],
                         dtype=np.float32)
        x = tf.placeholder(tf.float32, [3, None, 3], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=0, batch_axis=1, seq_lengths=[2, 3, 1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3],
                          [4, 5, 6], [4, 5, 6], [1, 1, 1],
                          [0, 0, 0], [7, 8, 9], [0, 0, 0]],
                         dtype=np.float32)
        x = tf.placeholder(tf.float32, [9, None], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=0, batch_axis=1, seq_lengths=[9, 9, 9])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val_shape = [5, 5, 7, 8, 9]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        x = tf.placeholder(tf.float32, [5, None, 7, 8, 9], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=0, batch_axis=1, seq_lengths=[5, 5, 5, 5, 5])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "where")
    def test_where(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.float32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.float32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        picks = tf.where(x > -1, true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

        tf.reset_default_graph()
        x_val = np.array(1, dtype=np.float32)
        true_result = np.array(100, dtype=np.float32)
        false_result = np.array(-111, dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        picks = tf.where(x > -1, true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "where")
    @check_target("rs6", "onnxruntime Where type limitation")
    def test_where_int32(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.int32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.int32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.int32)
        x = tf.placeholder(tf.int32, [None], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "where")
    def test_where_with_two_rank_input(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.float32)
        true_result = np.array([[111, 111], [222, 222], [333, 333], [444, 444], [555, 555],
                                [666, 666], [777, 777], [888, 888], [999, 999], [1000, 1000]],
                               dtype=np.float32)
        false_result = np.array([[-111, -111], [-222, -222], [-333, -333], [-444, -444],
                                 [-555, -555], [-666, -666], [-777, -777], [-888, -888],
                                 [-999, -999], [-1000, -1000]],
                                dtype=np.float32)
        x = tf.placeholder(tf.float32, [None], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "where")
    def test_where_with_two_rank_condition(self):
        x_val = np.array([[1, 2, -3, 4, -5, -6, -7, 8, 9, 0]], dtype=np.float32)
        true_result = np.array([[111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]],
                               dtype=np.float32)
        false_result = np.array([[-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000]],
                                dtype=np.float32)
        x = tf.placeholder(tf.float32, [1, 10], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "where")
    def test_where_with_three_rank_condition(self):
        x_val = np.array([[[1, 2, -3, 4, -5, -6, -7, 8, 9, 0]]], dtype=np.float32)
        true_result = np.array([[[111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]]],
                               dtype=np.float32)
        false_result = np.array([[[-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000]]],
                                dtype=np.float32)
        x = tf.placeholder(tf.float32, [1, 1, 10], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(8, "where")
    def test_where_scalar(self):
        x_val = np.array(6, dtype=np.float32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.float32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.float32)
        x = tf.placeholder(tf.float32, [], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @check_opset_min_version(9, "NonZero")
    @check_target("rs6", "onnxruntime Transpose type limitation")
    def test_where_with_cond_only(self):
        for np_type, tf_type in [(np.int32, tf.int32), (np.float32, tf.float32)]:
            x_val = np.random.randint(0, 2, size=[10, 20, 30]).astype(np_type)
            x = tf.placeholder(tf_type, shape=[None] * x_val.ndim, name=_TFINPUT)
            res = tf.where(x)
            _ = tf.identity(res, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})
            tf.reset_default_graph()

    @check_opset_min_version(6, "cast")
    def test_shape_int32(self):
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[None, 2, 3], name=_TFINPUT)
        x = tf.multiply(x, x)
        x = tf.shape(x, out_type=tf.int32)
        _ = tf.identity(x, name=_TFOUTPUT)
        kwargs = {"check_dtype": True}
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, **kwargs)

    @unittest.skipIf(get_test_config().is_onnxruntime_backend and get_test_config().opset < 7,
                     "mul-1, mul-6 not supported in onnxruntime. conversion is covered since opset6")
    def test_shape_int64(self):
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[None, 2, 3], name=_TFINPUT)
        x = tf.multiply(x, x)
        x = tf.shape(x, out_type=tf.int64)
        _ = tf.identity(x, name=_TFOUTPUT)
        kwargs = {"check_dtype": True}
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, **kwargs)

    # @check_opset_min_version(7, "broadcasting op")
    @unittest.skip("disable it for now, since fold const has bug")
    def test_softmax_cross_entropy_with_logits(self):
        num_class = 5
        data_shape = [100, num_class]
        for np_dtype, tf_dtype in zip([np.int32, np.int64], [tf.int32, tf.int64]):
            tf.reset_default_graph()
            label_val = np.random.randint(0, num_class - 1, data_shape).astype(np_dtype)
            logits_val = np.random.random(data_shape).astype(np.float32)

            label = tf.placeholder(tf_dtype, shape=data_shape, name=_TFINPUT)
            logits = tf.placeholder(tf.float32, shape=data_shape, name=_TFINPUT1)

            res1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
            _ = tf.identity(res1, name=_TFOUTPUT)

            self._run_test_case([_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val}, atol=1e-5)

    def test_sparse_softmax_cross_entropy_with_logits(self):
        num_class = 5
        label_val = np.array([3, 2, 0, 4]).astype(np.int32)
        logits_val = np.random.random((len(label_val), num_class)).astype(np.float32)

        label = tf.placeholder(tf.int32, shape=[None], name=_TFINPUT)
        logits = tf.placeholder(tf.float32, shape=[None, num_class], name=_TFINPUT1)

        res1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        _ = tf.identity(res1, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val})

    @check_target('rs6', 'SparseSoftmaxCrossEntropyWithLogits')
    def test_sparse_softmax_cross_entropy_with_logits_large_class(self):
        num_class = 30000
        label_val = np.array([3374, 2127, 10002, 48]).astype(np.int32)
        logits_val = np.random.random((len(label_val), num_class)).astype(np.float32)

        label = tf.placeholder(tf.int32, shape=[None], name=_TFINPUT)
        logits = tf.placeholder(tf.float32, shape=[None, num_class], name=_TFINPUT1)

        res = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        _ = tf.identity(res, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val}, rtol=1e-6)

    @skip_onnxruntime_backend("onnxruntime Slice did not supported BOOL")
    def test_matrix_band_part(self):
        input_val = np.random.randint(0, 666, (10, 15)).astype(np.int32)
        input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name=_TFINPUT)
        res = tf.matrix_band_part(input_x, -1, 0)
        res1 = tf.matrix_band_part(input_x, 0, -1)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    @skip_onnxruntime_backend("onnxruntime Slice did not supported BOOL.")
    def test_matrix_band_part_2(self):
        input_val = np.random.randint(0, 666, (1, 1)).astype(np.int32)
        input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name=_TFINPUT)
        res = tf.matrix_band_part(input_x, -1, 0)
        res1 = tf.matrix_band_part(input_x, 0, -1)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    def test_floordiv(self):
        input_val_1 = np.random.random_sample(100).astype(np.int32)
        input_val_2 = (np.random.random_sample(100) + 1).astype(np.int32)
        input_1 = tf.placeholder(dtype=tf.int32, shape=[None], name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.int32, shape=[None], name=_TFINPUT1)
        res = tf.floordiv(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

        tf.reset_default_graph()

        input_val_1 = np.random.random_sample(100).astype(np.float32)
        input_val_2 = (np.random.random_sample(100) + 1).astype(np.float32)
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None], name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None], name=_TFINPUT1)
        res = tf.floordiv(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

        tf.reset_default_graph()
        # test broadcasting
        input_val_1 = np.random.random_sample((10, 50)).astype(np.float32)
        input_val_2 = (np.random.random_sample(50) + 1).astype(np.float32)
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None] * input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None], name=_TFINPUT1)
        res = tf.floordiv(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

    def test_floormod(self):
        input_val_1 = 100 * np.random.random_sample(100).astype(np.int32)
        input_val_2 = (100 * np.random.random_sample(100) + 1).astype(np.int32)
        input_1 = tf.placeholder(dtype=tf.int32, shape=[None] * input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.int32, shape=[None] * input_val_2.ndim, name=_TFINPUT1)
        res = tf.floormod(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

        tf.reset_default_graph()

        input_val_1 = 100 * np.random.random_sample(100).astype(np.float32)
        input_val_2 = (100 * np.random.random_sample(100) + 1).astype(np.float32)
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None] * input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None] * input_val_2.ndim, name=_TFINPUT1)
        res = tf.floormod(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2}, rtol=1e-5)

        tf.reset_default_graph()
        # test broadcasting case
        input_val_1 = (50 * np.random.random_sample((10, 50)) + 1).astype(np.float32)
        input_val_2 = (50 * np.random.random_sample(50) + 1).astype(np.float32)
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None] * input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None] * input_val_2.ndim, name=_TFINPUT1)
        res = tf.floormod(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2}, rtol=1e-4)

    def test_logical_not(self):
        input_val = np.random.randint(0, 2, (10, 20)).astype(np.bool)
        input_x = tf.placeholder(dtype=tf.bool, shape=[None, None], name=_TFINPUT)
        res = tf.logical_not(input_x)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val})

    def test_reduce_all(self):
        input_val = np.random.randint(0, 2, (10, 20)).astype(np.bool)
        input_x = tf.placeholder(dtype=tf.bool, shape=[None] * input_val.ndim, name=_TFINPUT)
        res = tf.reduce_all(input_tensor=input_x, keepdims=False)
        res1 = tf.reduce_all(input_tensor=input_x, axis=[0], keepdims=False)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

        tf.reset_default_graph()

        input_val = np.random.randint(0, 2, (10, 20)).astype(np.bool)
        input_x = tf.placeholder(dtype=tf.bool, shape=[None] * input_val.ndim, name=_TFINPUT)
        res = tf.reduce_all(input_tensor=input_x, keepdims=True)
        res1 = tf.reduce_all(input_tensor=input_x, axis=[0], keepdims=True)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    def test_reduce_any(self):
        input_val = np.random.randint(0, 2, (10, 20)).astype(np.bool)
        input_x = tf.placeholder(dtype=tf.bool, shape=[None] * input_val.ndim, name=_TFINPUT)
        res = tf.reduce_any(input_tensor=input_x, keepdims=False)
        res1 = tf.reduce_any(input_tensor=input_x, axis=[0], keepdims=False)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

        tf.reset_default_graph()

        input_val = np.random.randint(0, 2, (10, 20)).astype(np.bool)
        input_x = tf.placeholder(dtype=tf.bool, shape=[None] * input_val.ndim, name=_TFINPUT)
        res = tf.reduce_any(input_tensor=input_x, keepdims=True)
        res1 = tf.reduce_any(input_tensor=input_x, axis=[0], keepdims=True)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    def test_zeros_like(self):
        input_val = np.random.random_sample([10, 20]).astype(np.float32)
        input_x = tf.placeholder(dtype=tf.float32, shape=[None] * input_val.ndim, name=_TFINPUT)
        res = tf.zeros_like(tensor=input_x)
        res1 = tf.zeros_like(tensor=input_x, dtype=tf.int32)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

        tf.reset_default_graph()

        input_val = np.random.random_sample([10, 20]).astype(np.int32)
        input_x = tf.placeholder(dtype=tf.int32, shape=[None] * input_val.ndim, name=_TFINPUT)
        res = tf.zeros_like(tensor=input_x)
        res1 = tf.zeros_like(tensor=input_x, dtype=tf.float32)
        _ = tf.identity(res, name=_TFOUTPUT)
        _ = tf.identity(res1, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: input_val})

    @check_opset_min_version(9, "is_nan")
    def test_isnan(self):
        # only compatible with dtype `float32`
        x_val1 = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32).reshape((2, 2))
        x_val3 = np.array([1.0, np.nan, -3.0, np.nan], dtype=np.float32).reshape((2, 2))
        for x_val in [x_val1, x_val2, x_val3]:
            x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
            x_ = tf.is_nan(x)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})
            tf.reset_default_graph()

    def test_ceil(self):
        x_val = np.array([-1.5, 1.2], dtype=np.float32)
        x = tf.placeholder(tf.float32, [2], name=_TFINPUT)
        x_ = tf.ceil(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_softplus(self):
        x_val = np.array([-1, 0, 1], dtype=np.float32)
        x = tf.placeholder(tf.float32, [3], name=_TFINPUT)
        x_ = tf.nn.softplus(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_softsign(self):
        x_val = np.array([-1, 0, 1], dtype=np.float32)
        x = tf.placeholder(tf.float32, [3], name=_TFINPUT)
        x_ = tf.nn.softsign(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_batch_to_spacend(self):
        block_size = [2, 2]
        crop = [[0, 1], [2, 1]]

        input_val = np.random.random_sample([40, 3, 5, 100]).astype(np.float32)
        input_x = tf.placeholder(dtype=tf.float32, shape=input_val.shape, name=_TFINPUT)  # NHWC
        _ = tf.batch_to_space_nd(input_x, block_size, crop, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val})

    def test_space_to_batchnd(self):
        block_size = [2, 2]
        pad = [[0, 1], [2, 1]]
        input_val = np.random.random_sample([40, 5, 7, 66]).astype(np.float32)
        input_x = tf.placeholder(dtype=tf.float32, shape=input_val.shape, name=_TFINPUT)  # NHWC
        _ = tf.space_to_batch_nd(input_x, block_size, pad, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val})

        tf.reset_default_graph()

        pad = [[0, 0], [1, 2]]
        input_val = np.random.random_sample([10, 6, 7, 66]).astype(np.float32)
        input_x = tf.placeholder(dtype=tf.float32, shape=input_val.shape, name=_TFINPUT)  # NHWC
        _ = tf.space_to_batch_nd(input_x, block_size, pad, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val})

    @check_opset_min_version(10, "is_inf")
    def test_isinf(self):
        x_types = [np.float32, np.float64]
        for x_type in x_types:
            x_val1 = np.array([1.0, -2.0, 3.0, -4.0], dtype=x_type)
            x_val2 = np.array([np.inf, np.inf, np.inf, np.inf], dtype=x_type).reshape((2, 2))
            x_val3 = np.array([1.0, np.inf, -3.0, np.inf, 5.0, np.inf, -7.0, np.inf], dtype=x_type).reshape((2, 2, 2))
            for x_val in [x_val1, x_val2, x_val3]:
                x = tf.placeholder(x_type, x_val.shape, name=_TFINPUT)
                x_ = tf.is_inf(x)
                _ = tf.identity(x_, name=_TFOUTPUT)
                self._run_test_case([_OUTPUT], {_INPUT: x_val})
                tf.reset_default_graph()

    @check_opset_min_version(10, "NonMaxSuppression")
    def test_non_max_suppression(self):
        box_num = 10
        boxes_val = np.random.random_sample([box_num, 4]).astype(np.float32)
        scores_val = np.random.random_sample([box_num]).astype(np.float32)
        boxes = tf.placeholder(tf.float32, shape=[None, 4], name=_TFINPUT)
        scores = tf.placeholder(tf.float32, shape=[None], name=_TFINPUT1)
        res1 = tf.image.non_max_suppression(boxes, scores, max_output_size=int(box_num / 2))
        res2 = tf.image.non_max_suppression(boxes, scores, max_output_size=0)
        _ = tf.identity(res1, name=_TFOUTPUT)
        _ = tf.identity(res2, name=_TFOUTPUT1)
        self._run_test_case([_OUTPUT, _OUTPUT1], {_INPUT: boxes_val, _INPUT1: scores_val})


if __name__ == '__main__':
    unittest_main()
