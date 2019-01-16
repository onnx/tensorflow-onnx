# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit tests using onnx backends."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from itertools import product
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase

# pylint: disable=missing-docstring,invalid-name,unused-argument


# we can override BACKEND and OPSET from the command line, but that is to late
# to change the behavior of annotation. If need, pick the backend here.
OPSET = Tf2OnnxBackendTestBase.OPSET
BACKEND = Tf2OnnxBackendTestBase.BACKEND
TARGET = Tf2OnnxBackendTestBase.TARGET

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


# pylint: disable=C0111


def make_xval(shape):
    x_val = np.arange(np.prod(shape)).astype("float32").reshape(shape)
    return x_val


def get_conv_getdata(kind=1):
    if kind == 0:
        # generate all combinations (costly)
        dims = [
            ("padding", ["SAME", "VALID"]),
            ("input_sizes", [[32, 35, 35, 288], [32, 17, 17, 1248], [1, 28, 28, 3], [32, 8, 8, 2048]]),
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
            ('SAME', [32, 35, 35, 288], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 288], [1, 2, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 288], [1, 1, 1, 1], [1, 1, 1, 1]),
            ('SAME', [32, 35, 35, 288], [1, 5, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 288], [1, 2, 5, 1], [1, 2, 2, 1]),
            ('SAME', [32, 35, 35, 288], [1, 2, 5, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 2, 2, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 2, 2, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 5, 1], [1, 2, 2, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 5, 1], [1, 1, 1, 1]),
            ('SAME', [1, 28, 28, 3], [1, 5, 2, 1], [1, 2, 2, 1]),
            ('SAME', [32, 8, 8, 2048], [1, 3, 3, 1], [1, 2, 2, 1]),
            ('SAME', [32, 8, 8, 2048], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('VALID', [32, 35, 35, 288], [1, 3, 3, 1], [1, 1, 1, 1]),
            ('VALID', [32, 35, 35, 288], [1, 2, 2, 1], [1, 2, 2, 1]),
        ]
        for idx, v in enumerate(data):
            yield (idx,) + v
    else:
        raise ValueError("kind not known")


def support_op_conversion_since(opset, op):
    return [OPSET < opset, op + " conversion is covered since opset " + str(opset)]


def support_op_with_target(target, op):
    return [target not in TARGET, op + " conversion is only supported with target " + str(target)]


def onnxruntime_check(op):
    if BACKEND not in ["onnxruntime"]:
        return (False, "")

    if op == "AveragePool":
        import onnxruntime as ort
        if ort.__version__ == "0.1.4":
            return (True, "Skip AveragePool for onnxruntime 0.1.4")

    support_since = {
        "Abs": 6, #  Abs-1
        "Add": 7, #  Add-1, Add-6
        "AveragePool": 7, #  AveragePool-1
        "Div": 7, #  Div-1, Div-6
        "Elu": 6, #  Elu-1
        "Exp": 6, #  Exp-1
        "Greater": 7, #  Greater-1
        "Log": 6, #  Log-1
        "Max": 6, #  Max-1
        "Min": 6, #  Min-1
        "Mul": 7, #  Mul-1, Mul-6
        "Neg": 6, #  Neg-1
        "Pow": 7, #  Pow-1
        "Reciprocal": 6, #  Reciprocal-1
        "Relu": 6, #  Relu-1
        "Sqrt": 6, #  Sqrt-1
        "Sub": 7, #  Sub-1, Sub-6
        "Tanh": 6, #  Tanh-1
    }
    if op not in support_since:
        return (False, "")

    cond = OPSET < support_since[op]
    message = op + " is supported by onnxruntime since opset " + str(support_since[op])

    return (cond, message)


class BackendTests(Tf2OnnxBackendTestBase):
    def _run_test_case(self, output_names_with_port, feed_dict, **kwargs):
        kwargs["convert_var_to_const"] = False
        kwargs["constant_fold"] = False
        self.run_test_case(feed_dict, [], output_names_with_port, **kwargs)

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

    @unittest.skipIf(*support_op_conversion_since(7, "trig"))
    def test_trig_ops(self):
        for op in [tf.sin, tf.cos, tf.tan, tf.asin, tf.acos, tf.atan]:
            tf.reset_default_graph()
            x_val = make_xval([3, 4])
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            op_ = op(x)
            _ = tf.identity(op_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-06)

    @unittest.skipIf(*support_op_conversion_since(9, "trigh"))
    def test_atrig_ops(self):
        for op in [tf.sinh, tf.cosh, tf.atanh, tf.asinh, tf.acosh]:
            tf.reset_default_graph()
            x_val = make_xval([3, 4])
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            op_ = op(x)
            _ = tf.identity(op_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "not supported correctly in caffe2")
    @unittest.skipIf(*support_op_conversion_since(7, "multinomial"))
    def test_multinomial(self):
        x_val = np.array([[10., 10.]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.multinomial(tf.log(x), 5, output_dtype=tf.int64)
        _ = tf.identity(op, name=_TFOUTPUT)

        # since returned indexes are random we can only check type and shape
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_value=False,
                            check_shape=True, check_dtype=True)


    @unittest.skipIf(BACKEND in ["caffe2"], "not supported correctly in caffe2")
    @unittest.skipIf(*support_op_conversion_since(7, "multinomial"))
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
        for p in get_conv_getdata():
            _, padding, x_shape, ksize, strides = p
            tf.reset_default_graph()
            x_val = make_xval(x_shape)
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            mp = tf.nn.max_pool(x, ksize, strides, padding=padding)
            _ = tf.identity(mp, name=_TFOUTPUT)

            self.log.debug(str(p))
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("AveragePool"))
    def test_avgpool(self):
        for p in get_conv_getdata(kind=0):
            _, padding, x_shape, ksize, strides = p
            tf.reset_default_graph()
            x_val = make_xval(x_shape)
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            mp = tf.nn.avg_pool(x, ksize, strides, padding=padding)
            _ = tf.identity(mp, name=_TFOUTPUT)

            self.log.debug(str(p))
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

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

    @unittest.skipIf(LooseVersion(tf.VERSION) < LooseVersion('1.7'), "tf only support dilation is 1 for now")
    def test_conv2d_7(self):
        x_shape = [1, 35, 35, 288]  # out: [1, 17, 17, 384]
        kernel_shape = [3, 3, 288, 384]
        strides = [1, 2, 2, 1]
        dilations = [1, 3, 3, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        self._conv_test(x_val, kernel_val, strides=strides, padding="VALID",
                        dilations=dilations, rtol=1e-05)

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

    @unittest.skipIf(*onnxruntime_check("Abs"))
    def test_abs(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.abs(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Add"))
    def test_const(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        y = tf.constant(x_val, name="y")
        _ = tf.add(x, y, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Add"))
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

    def test_placeholder_with_default(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        y = tf.constant(x_val, name="y")
        x = tf.placeholder_with_default(y, x_val.shape, name=_TFINPUT)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Add"))
    def test_add_bcast(self):
        x1_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x2_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32).reshape((2, 2, 2))
        # if we'd broadcast 2,2 to 2,1 onnxmsrt will fail
        x1 = tf.placeholder(tf.float32, x1_val.shape, name="input")
        x2 = tf.placeholder(tf.float32, x2_val.shape, name=_TFINPUT1)
        x_ = tf.add(x1, x2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x1_val, _INPUT1: x2_val})

    @unittest.skipIf(*onnxruntime_check("Add"))
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

    def test_matmul3(self):
        x_shape = [1, 12, 256, 64]
        x_val = np.arange(np.prod(x_shape)).astype("float32").reshape((x_shape))
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        y = tf.placeholder(tf.float32, x_shape, name=_TFINPUT1)
        x_ = tf.matmul(x, y, transpose_b=True)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: x_val}, rtol=1e-6)

    @unittest.skipIf(*onnxruntime_check("Sub"))
    def test_sub(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.subtract(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Mul"))
    def test_multiply(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.multiply(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Div"))
    def test_div(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.realdiv(x, x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Exp"))
    def test_exp(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.exp(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @unittest.skipIf(*onnxruntime_check("Log"))
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

    @unittest.skipIf(*support_op_with_target('rs6', 'GatherNd'))
    def test_gathernd(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        indices = np.array([[[0, 1], [1, 1]], [[1, 2], [0, 2]]], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather_nd(x, tf.constant(indices))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        indices = np.array([[0], [2], [4], [7]], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather_nd(x, tf.constant(indices))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_with_target('rs6', 'GatherNd'))
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

    @unittest.skipIf(BACKEND in ["caffe2"], "not supported in caffe2")
    @unittest.skipIf(*support_op_conversion_since(7, "tile"))
    def test_tile(self):
        x_val = np.array([[0, 1], [2, 3]], dtype=np.float32)
        multiple = tf.constant([2, 2])
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.tile(x, multiple)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Neg"))
    def test_neg(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.negative(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Mul"))
    def test_square(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.square(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Min"))
    def test_min(self):
        x_val1 = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32).reshape((2, 2))
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, x_val2.shape, name=_TFINPUT1)
        mi = tf.minimum(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @unittest.skipIf(BACKEND in ["caffe2"], "issue with broadcasting scalar")
    @unittest.skipIf(*onnxruntime_check("Sub"))
    def test_min_broadcast(self):
        # tests if the broadcast for min/max is working
        x_val1 = np.array([2.0, 16.0, 5.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.constant(x_val2, dtype=tf.float32, name='x2')
        mi = tf.minimum(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1})

    @unittest.skipIf(*onnxruntime_check("Add"))
    def test_logicaland(self):
        x_val1 = np.array([1, 0, 1, 1], dtype=np.bool).reshape((2, 2))
        x_val2 = np.array([0, 1, 1, 1], dtype=np.bool).reshape((2, 2))
        x1 = tf.placeholder(tf.bool, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.bool, [2, 2], name=_TFINPUT1)
        mi = tf.logical_and(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @unittest.skipIf(*onnxruntime_check("Greater"))
    def test_greater(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
        x1 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT1)
        mi = tf.greater(x1, x2)
        _ = tf.identity(mi, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val1, _INPUT1: x_val2})

    @unittest.skipIf(*onnxruntime_check("Greater"))
    def test_greater_unsupport_type(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.int32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.int32).reshape((2, 2))
        x1 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT1)
        mi = tf.greater(x1, x2)
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

    @unittest.skipIf(*support_op_conversion_since(6, "cast"))
    def test_reshape_int(self):
        x_val = np.array([1, 2, 3, 4], dtype=np.int32).reshape((2, 2))
        x = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
        shape = tf.constant([1, 4])
        x_ = tf.reshape(x, shape)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, check_shape=True)

    @unittest.skipIf(OPSET < 5 or BACKEND in ["onnxmsrtnext"], "since opset 5, broken in msrtnext")
    @unittest.skipIf(*support_op_conversion_since(6, "cast"))
    def test_reshape_dynamic(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        shape_val = np.array([4, 1], dtype=np.int32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        shape = tf.placeholder(tf.int32, shape_val.shape, name=_TFINPUT1)
        x_ = tf.reshape(x, shape)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: shape_val}, check_shape=True)

    @unittest.skipIf(*onnxruntime_check("Relu"))
    def test_relu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.relu(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND == "caffe2", "fails on caffe2 with dim issue")
    @unittest.skipIf(*onnxruntime_check("Mul"))
    def test_leaky_relu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.leaky_relu(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Elu"))
    def test_elu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.elu(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Tanh"))
    def test_tanh(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.tanh(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @unittest.skipIf(*onnxruntime_check("Max"))
    def test_relu6(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.relu6(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Sub"))
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

    @unittest.skipIf(*support_op_conversion_since(6, "cast"))
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

    @unittest.skipIf(*onnxruntime_check("Pow"))
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

    @unittest.skipIf(*support_op_conversion_since(9, "slice"))
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

    def test_split(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape(5, 30)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_, _, _ = tf.split(x0, [4, 15, 11], 1)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_reducesum(self):
        # not supported by onnx-caffe2
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reduce_sum(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*onnxruntime_check("Sqrt"))
    def test_sqrt(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.sqrt(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(7, "cast"))
    def test_range_const(self):
        x = tf.range(5)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})
        tf.reset_default_graph()

        x = tf.range(3, 3, 5)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})
        tf.reset_default_graph()

        x = tf.range(0, -5, -2)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})
        tf.reset_default_graph()

        x = tf.range(-5.0, 5.0, 1.5)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})
        tf.reset_default_graph()

        x = tf.range(2.5, 5.0, 10.0)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})

    # TODO: enable it later
    @unittest.skip("onnxruntime 0.1.3 has bug, this can pass with current latest onnxruntime")
    def test_range_non_const(self):
        x = tf.range(5.0)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})
        tf.reset_default_graph()

        x = tf.range(0, -5.0, -2)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})
        tf.reset_default_graph()

        x = tf.range(3.0, 3.0, 5)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {})
        tf.reset_default_graph()

        delta_val = np.array(1.5, dtype=np.float32)
        delta = tf.placeholder(tf.float32, shape=(), name=_TFINPUT)
        x = tf.range(-5.0, 5.0, delta)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: delta_val})
        tf.reset_default_graph()

        start_val = np.array(2.5, dtype=np.float32)
        start = tf.placeholder(tf.float32, shape=(), name=_TFINPUT)
        x = tf.range(start, 5.0, 10.0)
        _ = tf.identity(x, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: start_val})

    @unittest.skipIf(*onnxruntime_check("Sqrt"))
    def test_rsqrt(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.rsqrt(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, rtol=1e-05)

    @unittest.skipIf(*onnxruntime_check("Reciprocal"))
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

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
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

    @unittest.skip("")
    def test_slice1(self):
        # FIXME: only 1 dimension supported by caffe2 and msrt
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
        t1 = tf.constant([1, 0, 0], dtype=tf.int32)
        t2 = tf.constant([1, 1, 3], dtype=tf.int32)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.slice(x0, t1, t2)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "issue with broadcastnig scalar")
    @unittest.skipIf(*onnxruntime_check("Pow"))
    def test_pow_scalar(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
        e = np.array(2.0, dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.pow(x, tf.constant(e))
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND == "caffe2", "not supported correctly in caffe2")
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
            self.log.debug(str(p))
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND == "caffe2", "not supported correctly in caffe2")
    def test_pad_const(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        paddings = tf.constant([[1, 1,], [2, 2]], name="paddings")
        op = tf.pad(x, paddings, mode="CONSTANT", name="const_with_val", constant_values=999)

        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND == "caffe2", "not supported correctly in caffe2")
    def test_pad_reflect(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        paddings = tf.constant([[1, 1,], [2, 2]], name="paddings")
        op = tf.pad(x, paddings, mode="REFLECT", name="reflect")

        _ = tf.identity(op, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "not supported correctly in caffe2")
    def test_randomuniform(self):
        shape = tf.constant([2, 3], name="shape")
        x_ = tf.random_uniform(shape, name="rand", dtype=tf.float32)
        x_ = tf.identity(x_, name="output1")
        x_ = tf.identity(x_, name="output2")
        _ = tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case([_OUTPUT], {}, check_value=False, check_shape=True)

    @unittest.skip("")
    def test_randomuniform_int(self):
        shape = tf.constant([2, 3], name="shape")
        x_ = tf.random_uniform(shape, name="rand", dtype=tf.int32, maxval=10)
        x_ = tf.identity(x_, name="output1")
        x_ = tf.identity(x_, name="output2")
        _ = tf.identity(x_, name=_TFOUTPUT)
        # since results are random, compare the shapes only
        self._run_test_case([_OUTPUT], {}, check_value=False, check_shape=True)

    @unittest.skip("")
    def test_argminmax(self):
        # TODO: fails on onnxmsrt caffe2
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.argmin(x, axis=0)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(6, "cast"))
    def test_cast(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.cast(x, tf.int32)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(OPSET < 9, "supported since opset 9")
    def test_sign_int(self):
        x_val = np.array([1, 2, 0, -1, 0, -2], dtype=np.int).reshape((2, 3))
        x = tf.placeholder(tf.int32, [2, 3], name=_TFINPUT)
        x_ = tf.sign(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_sign(self):
        x_val = np.array([1.0, 2.0, 0.0, -1.0, 0.0, -2.0], dtype=np.float32).reshape((2, 3))
        x = tf.placeholder(tf.float32, [2, 3], name=_TFINPUT)
        x_ = tf.sign(x)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_onehot0(self):
        x_val = np.array([0, 1, 2], dtype=np.int32)
        depth = 5
        for axis in [-1, 0, 1]:
            tf.reset_default_graph()
            x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
            x_ = tf.one_hot(x, depth, on_value=5.0, axis=axis, off_value=1.0, dtype=tf.float32)
            _ = tf.identity(x_, name=_TFOUTPUT)
            self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skip("")
    def test_onehot1(self):
        # only rank 1 is currently implemented
        x_val = np.array([[0, 2], [1, -1]], dtype=np.int32)
        depth = 3
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x_ = tf.one_hot(x, depth, on_value=5.0, axis=-1, off_value=0.0, dtype=tf.float32)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_onehot2(self):
        x_val = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 2], dtype=np.int32)
        depth = 20
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x_ = tf.one_hot(x, depth, on_value=5.0, axis=-1, off_value=1.0, dtype=tf.float32)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "issue undefined dim 1")
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

    @unittest.skipIf(*support_op_conversion_since(6, "cast"))
    def test_topk(self):
        x_val = np.arange(3*2*3).astype("float32")
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        values, _ = tf.nn.top_k(x, 5, sorted=True)
        _ = tf.identity(values, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

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

    @unittest.skipIf(BACKEND in ["caffe2"], "Space2Depth not implemented, works on onnxmsrtnext")
    def test_space_to_depth(self):
        self._test_reorganize_data(tf.space_to_depth, [1, 2, 2, 1])

    @unittest.skipIf(BACKEND in ["caffe2"], "Space2Depth not implemented, works on onnxmsrtnext")
    def test_depth_to_space(self):
        self._test_reorganize_data(tf.depth_to_space, [1, 1, 1, 4])

    @unittest.skipIf(*support_op_conversion_since(6, "addn"))
    def test_addn(self):
        x_val = np.arange(3*2*3).astype("float32")
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.add_n([x, x, x])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "multiple dims not supported")
    def test_strided_slice1(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_strided_slice2(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [1, 0, 0], [2, 2, 3], [1, 1, 1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_strided_slice3(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[1:]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    def test_strided_slice4(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[:2]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "multiple dims not supported")
    def test_strided_slice5(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[:2, 0:1, 1:]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "multiple dims not supported")
    def test_strided_slice6(self):
        # example from here:
        # https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/ops/strided-slice
        x_val = np.arange(5*6).astype("float32").reshape(5, 6)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[2, :]
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(BACKEND in ["caffe2"], "fails with schema error")
    @unittest.skipIf(*support_op_conversion_since(7, "batchnorm"))
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

    @unittest.skipIf(BACKEND in ["caffe2"], "not correctly supported")
    @unittest.skipIf(*support_op_conversion_since(7, "resize_nearest_neighbor"))
    def test_resize_nearest_neighbor(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        x_new_size_ = tf.constant(x_new_size)
        x_ = tf.image.resize_nearest_neighbor(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(9, "resize_nearest_neighbor"))
    def test_resize_nearest_neighbor_with_non_const(self):
        x_shape = [3, 10, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape), dtype=np.float32).reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)

        x_new_size = np.array([20, 16]).astype(np.int32)
        x_new_size_ = tf.placeholder(shape=[None], dtype=tf.int32, name=_TFINPUT1)

        x_ = tf.image.resize_nearest_neighbor(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @unittest.skipIf(BACKEND in ["caffe2"], "not correctly supported")
    @unittest.skipIf(*support_op_conversion_since(7, "resize_bilinear"))
    def test_resize_bilinear(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        x_new_size_ = tf.constant(x_new_size)
        x_ = tf.image.resize_bilinear(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(9, "resize_bilinear"))
    def test_resize_bilinear_with_non_const(self):
        x_shape = [3, 10, 8, 5]
        x_val = np.arange(1, 1 + np.prod(x_shape), dtype=np.float32).reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)

        x_new_size = np.array([20, 16]).astype(np.int32)
        x_new_size_ = tf.placeholder(shape=[None], dtype=tf.int32, name=_TFINPUT1)

        x_ = tf.image.resize_bilinear(x, x_new_size_)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: x_new_size})

    @unittest.skipIf(*support_op_conversion_since(9, "fill"))
    def test_fill_float32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9.0)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(9, "fill"))
    def test_fill_int32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("int32").reshape(x_shape)
        x0 = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(7, "fill"))
    def test_fill7_float32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9.0)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(7, "fill"))
    def test_fill7_int32(self):
        x_shape = [1, 15, 20, 2]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("int32").reshape(x_shape)
        x0 = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x1 = tf.fill(x_val.shape, 9)
        x2 = tf.add(x0, x1)
        _ = tf.identity(x2, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(7, "div"))
    def test_tf_div(self):
        # pylint: disable=E0001
        from tensorflow.python.ops.gen_math_ops import div
        shape = 1000
        # test floating data
        x_val = (np.random.sample(shape)+1e-6).astype(np.float32)
        y_val = (np.random.sample(shape)+1e-6).astype(np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        y = tf.placeholder(tf.float32, y_val.shape, name=_TFINPUT1)
        output = div(x, y, name=_TFOUTPUT)
        assert output.op.type == "Div"
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

        tf.reset_default_graph()
        # test integer data
        x_val = (100*np.random.sample(shape)+1).astype(np.int32)
        y_val = (100*np.random.sample(shape)+1).astype(np.int32)
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        y = tf.placeholder(tf.int32, y_val.shape, name=_TFINPUT1)
        output = div(x, y, name=_TFOUTPUT)
        assert output.op.type == "Div"
        self._run_test_case([_OUTPUT], {_INPUT: x_val, _INPUT1: y_val})

    @unittest.skipIf(*support_op_conversion_since(7, "erf"))
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

    # @unittest.skipIf(OPSET < 8, "supported with opset 8 or better")
    @unittest.skip("FIXME: the newest onnxruntime wheel hasn't been published to PYPI, so scan op is not supported")
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
        x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[3]*9)
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val_shape = [5, 5, 7, 8, 9]
        x_val = np.random.randint(0, 100, x_val_shape).astype(np.float32)
        x = tf.placeholder(tf.float32, [None, 5, 7, 8, 9], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=1, batch_axis=0, seq_lengths=[5, 5, 5, 5, 5])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    # @unittest.skipIf(OPSET < 8, "supported with opset 8 or better")
    @unittest.skip("FIXME: the newest onnxruntime wheel hasn't been published to PYPI, so scan op is not supported")
    def test_reverse_sequence_time_major(self):
        x_val = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                          [[4, 5, 6], [4, 5, 6], [0, 0, 0]],
                          [[0, 0, 0], [7, 8, 9], [0, 0, 0]]
                         ],
                         dtype=np.float32)
        x = tf.placeholder(tf.float32, [3, None, 3], name=_TFINPUT)
        x_ = tf.reverse_sequence(x, seq_axis=0, batch_axis=1, seq_lengths=[2, 3, 1])
        _ = tf.identity(x_, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})
        tf.reset_default_graph()

        x_val = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3],
                          [4, 5, 6], [4, 5, 6], [1, 1, 1],
                          [0, 0, 0], [7, 8, 9], [0, 0, 0]
                         ],
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

    # @unittest.skipIf(OPSET < 8, "supported with opset 8 or better")
    @unittest.skip("FIXME: the newest onnxruntime wheel hasn't been published to PYPI, so Select op is not supported")
    def test_where(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.int32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.int32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.int32)
        x = tf.placeholder(tf.int32, [None], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(8, "where"))
    def test_where_with_two_rank_input(self):
        x_val = np.array([1, 2, -3, 4, -5, -6, -7, 8, 9, 0], dtype=np.int32)
        true_result = np.array([[111, 111], [222, 222], [333, 333], [444, 444], [555, 555],
                                [666, 666], [777, 777], [888, 888], [999, 999], [1000, 1000]],
                               dtype=np.int32)
        false_result = np.array([[-111, -111], [-222, -222], [-333, -333], [-444, -444],
                                 [-555, -555], [-666, -666], [-777, -777], [-888, -888],
                                 [-999, -999], [-1000, -1000]],
                                dtype=np.int32)
        x = tf.placeholder(tf.int32, [None], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(8, "where"))
    def test_where_with_two_rank_condition(self):
        x_val = np.array([[1, 2, -3, 4, -5, -6, -7, 8, 9, 0]], dtype=np.int32)
        true_result = np.array([[111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]],
                               dtype=np.int32)
        false_result = np.array([[-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000]],
                                dtype=np.int32)
        x = tf.placeholder(tf.int32, [1, 10], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(8, "where"))
    def test_where_with_three_rank_condition(self):
        x_val = np.array([[[1, 2, -3, 4, -5, -6, -7, 8, 9, 0]]], dtype=np.int32)
        true_result = np.array([[[111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]]],
                               dtype=np.int32)
        false_result = np.array([[[-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000]]],
                                dtype=np.int32)
        x = tf.placeholder(tf.int32, [1, 1, 10], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(8, "where"))
    def test_where_scalar(self):
        x_val = np.array(6, dtype=np.int32)
        true_result = np.array([111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
                               dtype=np.int32)
        false_result = np.array([-111, -222, -333, -444, -555, -666, -777, -888, -999, -1000],
                                dtype=np.int32)
        x = tf.placeholder(tf.int32, [], name=_TFINPUT)
        picks = tf.where(tf.greater_equal(x, 0), true_result, false_result)
        _ = tf.identity(picks, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: x_val})

    @unittest.skipIf(*support_op_conversion_since(6, "cast"))
    def test_shape_int32(self):
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[None, 2, 3], name=_TFINPUT)
        x = tf.multiply(x, x)
        x = tf.shape(x, out_type=tf.int32)
        _ = tf.identity(x, name=_TFOUTPUT)
        kwargs = {"check_dtype": True}
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, **kwargs)

    @unittest.skipIf(BACKEND in ["onnxruntime"] and OPSET < 7,
                     "mul-1, mul-6 not supported in onnxruntime. conversion is covered since opset6")
    def test_shape_int64(self):
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=[None, 2, 3], name=_TFINPUT)
        x = tf.multiply(x, x)
        x = tf.shape(x, out_type=tf.int64)
        _ = tf.identity(x, name=_TFOUTPUT)
        kwargs = {"check_dtype": True}
        self._run_test_case([_OUTPUT], {_INPUT: x_val}, **kwargs)

    def test_sparse_softmax_cross_entropy_with_logits(self):
        num_class = 5
        label_val = np.array([3, 2, 0, 4]).astype(np.int32)
        logits_val = np.random.random((len(label_val), num_class)).astype(np.float32)

        label = tf.placeholder(tf.int32, shape=[None], name=_TFINPUT)
        logits = tf.placeholder(tf.float32, shape=[None, num_class], name=_TFINPUT1)

        res1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        _ = tf.identity(res1, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val})

    @unittest.skipIf(*support_op_with_target('rs6', 'SparseSoftmaxCrossEntropyWithLogits'))
    def test_sparse_softmax_cross_entropy_with_logits_large_class(self):
        num_class = 30000
        label_val = np.array([3374, 2127, 10002, 48]).astype(np.int32)
        logits_val = np.random.random((len(label_val), num_class)).astype(np.float32)

        label = tf.placeholder(tf.int32, shape=[None], name=_TFINPUT)
        logits = tf.placeholder(tf.float32, shape=[None, num_class], name=_TFINPUT1)

        res = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        _ = tf.identity(res, name=_TFOUTPUT)

        self._run_test_case([_OUTPUT], {_INPUT: label_val, _INPUT1: logits_val})

    @unittest.skip("TODO: add a common utility for onnxruntime version check in another PR")
    def test_matrix_band_part(self):
        input_val = np.random.randint(0, 666, (10, 15)).astype(np.int32)
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
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None]*input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None], name=_TFINPUT1)
        res = tf.floordiv(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

    def test_floormod(self):
        input_val_1 = 100*np.random.random_sample(100).astype(np.int32)
        input_val_2 = (100*np.random.random_sample(100) + 1).astype(np.int32)
        input_1 = tf.placeholder(dtype=tf.int32, shape=[None]*input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.int32, shape=[None]*input_val_2.ndim, name=_TFINPUT1)
        res = tf.floormod(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2})

        tf.reset_default_graph()

        input_val_1 = 100*np.random.random_sample(100).astype(np.float32)
        input_val_2 = (100*np.random.random_sample(100) + 1).astype(np.float32)
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None]*input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None]*input_val_2.ndim, name=_TFINPUT1)
        res = tf.floormod(input_1, input_2)
        _ = tf.identity(res, name=_TFOUTPUT)
        self._run_test_case([_OUTPUT], {_INPUT: input_val_1, _INPUT1: input_val_2}, rtol=1e-5)

        tf.reset_default_graph()
        # test broadcasting case
        input_val_1 = (50*np.random.random_sample((10, 50)) + 1).astype(np.float32)
        input_val_2 = (50*np.random.random_sample(50) + 1).astype(np.float32)
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None]*input_val_1.ndim, name=_TFINPUT)
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None]*input_val_2.ndim, name=_TFINPUT1)
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
        input_x = tf.placeholder(dtype=tf.bool, shape=[None]*input_val.ndim, name=_TFINPUT)
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
        input_x = tf.placeholder(dtype=tf.bool, shape=[None]*input_val.ndim, name=_TFINPUT)
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


if __name__ == '__main__':
    Tf2OnnxBackendTestBase.trigger(BackendTests)
