# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tempfile
import unittest
from collections import namedtuple
from itertools import product

import numpy as np
import tensorflow as tf
import tf2onnx.utils
from onnx import helper
from tf2onnx.tfonnx import process_tf_graph

TMPPATH = tempfile.mkdtemp()

# we can override BACKEND and OPSET from the command line, but that is to late
# to change the behavior of annotation. If need, pick the backend here.
OPSET = 7

BACKEND = "caffe2"
# BACKEND = "onnxmsrt"
# BACKEND = "onnxmsrtnext"
# BACKEND = "onnx-tensorflow"

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
_TFOUTPUT = "output"
_OUTPUT = "output:0"
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


class Tf2OnnxBackendTests(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        tf.reset_default_graph()
        # reset name generation on every test
        tf2onnx.utils.INTERNAL_NAME = 1
        np.random.seed(1)  # Make it reproducible.

        arg = namedtuple("Arg", "input inputs outputs verbose continue_on_error")
        self._args0 = arg(input="test", inputs=[], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args1 = arg(input="test", inputs=[_INPUT], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args2 = arg(input="test", inputs=[_INPUT, _INPUT1], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args3 = arg(input="test", inputs=[_INPUT, _INPUT1, "prob:0"], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args4 = arg(input="test", inputs=[_INPUT, _INPUT1], outputs=[_OUTPUT, _OUTPUT1],
                          verbose=False, continue_on_error=False)

    @staticmethod
    def assertAllClose(expected, actual, **kwargs):
        np.testing.assert_allclose(expected, actual, **kwargs)

    @staticmethod
    def run_onnxcaffe2(onnx_graph, inputs):
        """Run test against caffe2 backend."""
        import caffe2.python.onnx.backend
        prepared_backend = caffe2.python.onnx.backend.prepare(onnx_graph)
        results = prepared_backend.run(inputs)
        return results[0]

    @staticmethod
    def run_onnxmsrt(onnx_graph, inputs, output_names, test_name):
        """Run test against msrt backend."""
        import lotus
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())
        m = lotus.ModelExecutor(model_path)
        results = m.run(output_names, inputs)
        return results[0]

    @staticmethod
    def run_onnxmsrtnext(onnx_graph, inputs, output_names, test_name):
        """Run test against msrt-next backend."""
        import lotus
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        # print(model_path)
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())
        m = lotus.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results[0]

    @staticmethod
    def run_onnxcntk(onnx_graph, inputs, test_name):
        """Run test against cntk backend."""
        import cntk as C
        print(test_name)
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())
        z = C.Function.load(model_path, format=C.ModelFormat.ONNX)
        input_args = {}
        for arg in z.arguments:
            input_args[arg] = inputs[arg.name]
        results = z.eval(input_args)
        return results

    @staticmethod
    def run_onnxtensorflow(onnx_graph, inputs):
        """Run test against onnx-tensorflow backend."""
        import onnx_tf.backend
        prepared_backend = onnx_tf.backend.prepare(onnx_graph)
        results = prepared_backend.run(inputs)
        return results

    def _run_backend(self, g, args, input_dict, expected):
        model_proto = g.make_model("test", args.inputs, args.outputs)
        if BACKEND == "onnxmsrt":
            y = self.run_onnxmsrt(model_proto, input_dict, args.outputs, self._testMethodName)
        elif BACKEND == "onnxmsrtnext":
            y = self.run_onnxmsrtnext(model_proto, input_dict, args.outputs, self._testMethodName)
        elif BACKEND == "cntk":
            y = self.run_onnxcntk(model_proto, input_dict, self._testMethodName)
        elif BACKEND == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        elif BACKEND == "onnxnumpy":
            y = self.run_onnxnumpy(model_proto, input_dict)
            y = y[args.outputs[0]]
        elif BACKEND == "onnx-tensorflow":
            y = self.run_onnxtensorflow(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y

    def _run(self, output, tf_dict, onnx_dict):
        with tf.Session() as sess:
            expected = sess.run(output, feed_dict=tf_dict)
            g = process_tf_graph(sess.graph)
            actual = self._run_backend(g, self._args1, onnx_dict, expected)
        return actual, expected

    def _test_expand_dims(self, idx):
        tf.reset_default_graph()
        x_val = make_xval([3, 4])
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.expand_dims(x, idx)
        output = tf.identity(op, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_expand_dims(self):
        for i in [-1, 0, 1, -2]:
            self._test_expand_dims(i)

    def test_trig_ops(self):
        for op in [tf.sin, tf.cos, tf.tan, tf.asin, tf.acos, tf.atan]:
            tf.reset_default_graph()
            x_val = make_xval([3, 4])
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            op_ = op(x)
            output = tf.identity(op_, name=_TFOUTPUT)
            actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
            self.assertAllClose(expected, actual, rtol=1e-06)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "not supported correctly in caffe2")
    def test_multinomial(self):
        x_val = np.array([[10., 10.]], dtype=np.float32)
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.multinomial(tf.log(x), 5, output_dtype=tf.int64)
        output = tf.identity(op, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        # since returned indexes are random we can only check type and shape
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "not supported correctly in caffe2")
    def test_multinomial1(self):
        shape = [2, 10]
        x_val = np.ones(np.prod(shape)).astype("float32").reshape(shape)
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.multinomial(x, 2, output_dtype=tf.int64)
        output = tf.identity(op, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        # since returned indexes are random we can only check type and shape
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)

    def test_maxppol(self):
        for p in get_conv_getdata():
            idx, padding, x_shape, ksize, strides = p
            tf.reset_default_graph()
            x_val = make_xval(x_shape)
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            mp = tf.nn.max_pool(x, ksize, strides, padding=padding)
            output = tf.identity(mp, name=_TFOUTPUT)
            actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
            self.assertAllClose(expected, actual, err_msg=str(p))

    def test_avgppol(self):
        for p in get_conv_getdata(kind=0):
            idx, padding, x_shape, ksize, strides = p
            tf.reset_default_graph()
            x_val = make_xval(x_shape)
            x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
            mp = tf.nn.avg_pool(x, ksize, strides, padding=padding)
            output = tf.identity(mp, name=_TFOUTPUT)
            actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
            self.assertAllClose(expected, actual, err_msg=str(p))

    def _conv_test(self, x_val, w, strides=None, padding="VALID", dilations=None):
        if strides is None:
            strides = _STRIDE1x1
        if dilations is None:
            dilations = [1, 1, 1, 1]
        tf.reset_default_graph()
        kernel = tf.constant(w, dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.conv2d(x, kernel, strides=strides, padding=padding, dilations=dilations)
        output = tf.identity(conv, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        return actual, expected

    def test_conv2d_1(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape(_KERNEL3x3)
        expected, actual = self._conv_test(x_val, w)
        self.assertAllClose(expected, actual)

    def test_conv2d_2(self):
        x_val = np.array([[4, 3, 1, 0],
                          [2, 1, 0, 1],
                          [1, 2, 4, 1],
                          [3, 1, 0, 2]], dtype=np.float32).reshape([1, 4, 4, 1])
        w = np.array([[1, 0, 1],
                      [2, 1, 0],
                      [0, 0, 1]], dtype=np.float32).reshape(_KERNEL3x3)
        expected, actual = self._conv_test(x_val, w)
        self.assertAllClose(expected, actual)

    def test_conv2d_3(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape(_KERNEL3x3)
        expected, actual = self._conv_test(x_val, w)
        self.assertAllClose(expected, actual)

    def test_conv2d_4(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        w = np.random.random_sample(_KERNEL3x3).astype(np.float32)
        expected, actual = self._conv_test(x_val, w, padding="SAME")
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_conv2d_5(self):
        x_val = make_xval((1, 1, 5, 5)).transpose(NCHW_TO_NHWC)
        kernel_shape = [3, 3, 1, 2]
        w = np.random.random_sample(kernel_shape).astype(np.float32)
        expected, actual = self._conv_test(x_val, w, padding="SAME")
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_conv2d_6(self):
        x_shape = [1, 35, 35, 288]  # out: [1, 17, 17, 384]
        kernel_shape = [3, 3, 288, 384]
        strides = [1, 2, 2, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        expected, actual = self._conv_test(x_val, kernel_val, strides=strides, padding="VALID")
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_conv2d_7(self):
        x_shape = [1, 35, 35, 288]  # out: [1, 17, 17, 384]
        kernel_shape = [3, 3, 288, 384]
        strides = [1, 2, 2, 1]
        dilations = [1, 3, 3, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        expected, actual = self._conv_test(x_val, kernel_val, strides=strides, padding="VALID", dilations=dilations)
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_conv2d_transpose(self):
        x_shape = [2, 6, 4, 3]
        output_shape = [2, 13, 9, 2]
        kernel_shape = [3, 3, 2, 3]
        strides = [1, 2, 2, 1]
        x_val = make_xval(x_shape)
        kernel_val = make_xval(kernel_shape)
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, shape=x_shape, name=_TFINPUT)
            f = tf.constant(kernel_val, name="kernel", dtype=tf.float32)
            conv = tf.nn.conv2d_transpose(x, f, output_shape, strides=strides, padding="VALID")
            output = tf.identity(conv, name=_TFOUTPUT)
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={x: x_val})
            g = process_tf_graph(sess.graph)
            actual = self._run_backend(g, self._args1, {_INPUT: x_val}, expected)
            self.assertAllClose(expected, actual, rtol=1e-05)

    def test_depthwiseconv_0(self):
        x_shape = [1, 3, 4, 3]
        kernel_shape = [3, 3, 3, 3]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        output = tf.identity(conv, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
        self.assertAllClose(expected, actual, rtol=0.08)

    def test_depthwiseconv_1(self):
        x_shape = [1, 112, 112, 32]
        kernel_shape = [3, 3, 32, 1]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        conv = tf.nn.depthwise_conv2d(x, kernel, strides=_STRIDE1x1, padding='VALID')
        output = tf.identity(conv, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
        self.assertAllClose(expected, actual, rtol=0.08)

    @unittest.skip
    def test_lrn(self):
        # FIXME: numerical results are not correct
        x_shape = [1, 3, 4, 3]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, shape=x_val.shape, name=_TFINPUT)
        op = tf.nn.local_response_normalization(x_val)
        output = tf.identity(op, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_abs(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.abs(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_const(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        y = tf.constant(x_val, name="y")
        output = tf.add(x, y, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_add(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.add(x, x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_placeholder(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        output = tf.identity(x, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_placeholder_with_default(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        y = tf.constant(x_val, name="y")
        x = tf.placeholder_with_default(y, x_val.shape, name=_TFINPUT)
        output = tf.identity(x, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_add_bcast(self):
        x1_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x2_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32).reshape((2, 2, 2))
        # if we'd broadcast 2,2 to 2,1 onnxmsrt will fail
        x1 = tf.placeholder(tf.float32, x1_val.shape, name="input")
        x2 = tf.placeholder(tf.float32, x2_val.shape, name=_TFINPUT1)
        x_ = tf.add(x1, x2)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x1: x1_val, x2: x2_val}, {_INPUT: x1_val, _INPUT1: x2_val})
        self.assertAllClose(expected, actual)

    def test_add_bcast1(self):
        # example taken from onnx doc
        x1_val = np.random.randn(3, 4, 5).astype(np.float32)
        x2_val = np.random.randn(5).astype(np.float32)
        x1 = tf.placeholder(tf.float32, x1_val.shape, name="input")
        x2 = tf.placeholder(tf.float32, x2_val.shape, name=_TFINPUT1)
        x_ = tf.add(x1, x2)
        output = tf.identity(x_, name=_TFOUTPUT)
        expected = np.add(x1_val, x2_val)
        actual, tf_actual = self._run(output, {x1: x1_val, x2: x2_val}, {_INPUT: x1_val, _INPUT1: x2_val})
        self.assertAllClose(expected, tf_actual)
        self.assertAllClose(expected, actual)

    def test_matmul(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.matmul(x, x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_sub(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.subtract(x, x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_multiply(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.multiply(x, x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_div(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.realdiv(x, x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_exp(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.exp(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_log(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.log(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_gather(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        idx = np.array([1, 0, 2], dtype=np.int32)
        idx_flattened = np.array([i * x_val.shape[1] + idx for i in range(0, x_val.shape[0])])
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.gather(tf.reshape(x, [-1]), tf.constant(idx_flattened))
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "not supported in caffe2")
    def test_tile(self):
        x_val = np.array([[0, 1], [2, 3]], dtype=np.float32)
        multiple = tf.constant([2, 2])
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.tile(x, multiple)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_neg(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.negative(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_square(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.square(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_min(self):
        x_val1 = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32).reshape((2, 2))
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, x_val2.shape, name=_TFINPUT1)
        mi = tf.minimum(x1, x2)
        output = tf.identity(mi, name=_TFOUTPUT)
        actual, expected = self._run(output, {x1: x_val1, x2: x_val2}, {_INPUT: x_val1, _INPUT1: x_val2, })
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "issue with broadcastnig scalar")
    def test_min_broadcast(self):
        # tests if the broadcast for min/max is working
        x_val1 = np.array([2.0, 16.0, 5.0, 1.6], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([4.0], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.constant(x_val2, dtype=tf.float32, name='x2')
        mi = tf.minimum(x1, x2)
        output = tf.identity(mi, name=_TFOUTPUT)
        actual, expected = self._run(output, {x1: x_val1}, {_INPUT: x_val1})
        self.assertAllClose(expected, actual)

    def test_logicaland(self):
        x_val1 = np.array([1, 0, 1, 1], dtype=np.bool).reshape((2, 2))
        x_val2 = np.array([0, 1, 1, 1], dtype=np.bool).reshape((2, 2))
        x1 = tf.placeholder(tf.bool, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.bool, [2, 2], name=_TFINPUT1)
        mi = tf.logical_and(x1, x2)
        output = tf.identity(mi, name=_TFOUTPUT)
        actual, expected = self._run(output, {x1: x_val1, x2: x_val2}, {_INPUT: x_val1, _INPUT1: x_val2})
        self.assertAllClose(expected, actual)

    def test_greater(self):
        x_val1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
        x_val2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
        x1 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT1)
        mi = tf.greater(x1, x2)
        output = tf.identity(mi, name=_TFOUTPUT)
        actual, expected = self._run(output, {x1: x_val1, x2: x_val2}, {_INPUT: x_val1, _INPUT1: x_val2})
        self.assertAllClose(expected, actual)

    def test_sequeeze(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2, 1))
        x = tf.placeholder(tf.float32, [2, 2, 1], name=_TFINPUT)
        x_ = tf.squeeze(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_transpose(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape((2, 3))
        x = tf.placeholder(tf.float32, [2, 3], name=_TFINPUT)
        x_ = tf.transpose(x)  # perm=[1,0])
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_reshape(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        shape = tf.constant([1, 4])
        x_ = tf.reshape(x, shape)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertEqual(expected.shape, actual.shape)
        self.assertAllClose(expected, actual)

    def test_reshape_int(self):
        x_val = np.array([1, 2, 3, 4], dtype=np.int32).reshape((2, 2))
        x = tf.placeholder(tf.int32, [2, 2], name=_TFINPUT)
        shape = tf.constant([1, 4])
        x_ = tf.reshape(x, shape)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertEqual(expected.shape, actual.shape)
        self.assertAllClose(expected, actual)

    @unittest.skipIf(OPSET < 5 or BACKEND in ["onnxmsrtnext"], "since opset 5, broken in msrtnext")
    def test_reshape_dynamic(self):
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        shape = tf.constant([4, 1])
        x_ = tf.reshape(x, tf.transpose(shape))
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertEqual(expected.shape, actual.shape)
        self.assertAllClose(expected, actual)

    def test_relu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.relu(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND == "caffe2", "fails on caffe2 with dim issue")
    def test_leaky_relu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.leaky_relu(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_elu(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.elu(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_tanh(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.tanh(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_relu6(self):
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.nn.relu6(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_concat(self):
        x_val1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x_val2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        x_val3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.float32)
        x1 = tf.placeholder(tf.float32, x_val1.shape, name=_TFINPUT)
        x2 = tf.placeholder(tf.float32, x_val2.shape, name=_TFINPUT1)
        x3 = tf.placeholder(tf.float32, x_val3.shape, name="input3")
        x_ = tf.concat([x1, x2, x3], 0)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x1: x_val1, x2: x_val2, x3: x_val3},
                                     {_INPUT: x_val1, _INPUT1: x_val2, "input3:0": x_val3})
        self.assertAllClose(expected, actual)

    def test_pow(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
        e = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.pow(x, tf.constant(e))
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_embedding_lookup(self):
        x_val1 = np.array([[1]], dtype=np.int32)
        x_val2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        t = tf.constant(x_val2)
        x = tf.placeholder(tf.int32, x_val1.shape, name=_TFINPUT)
        x_ = tf.nn.embedding_lookup(t, x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val1}, {_INPUT: x_val1})
        self.assertAllClose(expected, actual)

    def test_slice(self):
        x_val = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        t1 = tf.constant([0, 1], dtype=tf.int32)
        t2 = tf.constant([2, 2], dtype=tf.int32)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.slice(x0, t1, t2)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x0: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_split(self):
        x_val = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape(5, 30)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_, _, _ = tf.split(x0, [4, 15, 11], 1)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x0: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_reducesum(self):
        # not supported by onnx-caffe2
        x_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reduce_sum(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_sqrt(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.sqrt(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_rsqrt(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.rsqrt(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_reciprocal(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reciprocal(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual, rtol=1e-04)

    def test_reducemax(self):
        # not supported by onnx-caffe2
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.reduce_max(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual, rtol=1e-05)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_reduceprod(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reduce_prod(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_reducemean(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.reduce_mean(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skip
    def test_slice1(self):
        # FIXME: only 1 dimension supported by caffe2 and msrt
        x_val = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
        t1 = tf.constant([1, 0, 0], dtype=tf.int32)
        t2 = tf.constant([1, 1, 3], dtype=tf.int32)
        x0 = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.slice(x0, t1, t2)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x0: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2"], "issue with broadcastnig scalar")
    def test_pow_scalar(self):
        x_val = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
        e = np.array(2.0, dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.pow(x, tf.constant(e))
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND == "caffe2", "not supported correctly in caffe2")
    def test_pad(self):
        params = [
            ("CONSTANT", [[1, 1], [2, 2]], [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]),
            ("CONSTANT", [[0, 0], [3, 3], [3, 3], [0, 0]],  np.random.randn(1, 3, 4, 5).astype(np.float32)),
        ]
        for p in params:
            tf.reset_default_graph()
            mode, pad, xv = p
            x_val = np.array(xv, dtype=np.float32)
            x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
            paddings = tf.constant(pad)
            op = tf.pad(x, paddings, mode)
            output = tf.identity(op, name=_TFOUTPUT)
            actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
            self.assertAllClose(expected, actual, err_msg=str(p))

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "not supported correctly in caffe2")
    def test_randomuniform(self):
        shape = tf.constant([2, 3], name="shape")
        x_ = tf.random_uniform(shape, name="rand", dtype=tf.float32)
        x_ = tf.identity(x_, name="output1")
        x_ = tf.identity(x_, name="output2")
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {}, {})
        # since results are random, compare the shapes only
        self.assertAllClose(expected.shape, actual.shape)

    @unittest.skip
    def test_randomuniform_int(self):
        shape = tf.constant([2, 3], name="shape")
        x_ = tf.random_uniform(shape, name="rand", dtype=tf.int32, maxval=10)
        x_ = tf.identity(x_, name="output1")
        x_ = tf.identity(x_, name="output2")
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {}, {})
        # since results are random, compare the shapes only
        self.assertAllClose(expected.shape, actual.shape)

    @unittest.skip
    def test_argminmax(self):
        # TODO: fails on onnxmsrt caffe2
        x_val = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.argmin(x, axis=0)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_cast(self):
        x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
        x = tf.placeholder(tf.float32, [2, 2], name=_TFINPUT)
        x_ = tf.cast(x, tf.int32)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_onehot0(self):
        x_val = np.array([0, 1, 2], dtype=np.int32)
        depth = 5
        for axis in [-1, 0, 1]:
            tf.reset_default_graph()
            x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
            x_ = tf.one_hot(x, depth, on_value=5.0, axis=axis, off_value=1.0, dtype=tf.float32)
            output = tf.identity(x_, name=_TFOUTPUT)
            actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
            self.assertAllClose(expected, actual)

    @unittest.skip
    def test_onehot1(self):
        # only rank 1 is currently implemented
        x_val = np.array([[0, 2], [1, -1]], dtype=np.int32)
        depth = 3
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x_ = tf.one_hot(x, depth, on_value=5.0, axis=-1, off_value=0.0, dtype=tf.float32)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_onehot2(self):
        x_val = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 2], dtype=np.int32)
        depth = 20
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x_ = tf.one_hot(x, depth, on_value=5.0, axis=-1, off_value=1.0, dtype=tf.float32)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2"], "issue undefined dim 1")
    def test_flatten0(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, [None, 3, 3], name=_TFINPUT)
        x_ = tf.layers.flatten(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_flatten1(self):
        x_val = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, [1, 3, 3], name=_TFINPUT)
        x_ = tf.layers.flatten(x)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_cancel_transpose(self):
        x_val = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.identity(x, _TFINPUT)
        x_ = tf.transpose(x_, perm=NHWC_TO_NCHW)
        x_ = tf.transpose(x_, perm=NCHW_TO_NHWC)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_topk(self):
        x_val = np.arange(3*2*3).astype("float32")
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        values, indices = tf.nn.top_k(x, 5, sorted=True)
        output = tf.identity(values, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_stack_axis(self):
        for axis in [0, 1]:
            tf.reset_default_graph()
            x_val = [np.random.randn(3, 4).astype("float32") for _ in range(10)]
            x = [tf.constant(x_val[i], dtype=tf.float32) for i in range(10)]
            x_ = tf.stack(x, axis=axis)
            output = tf.identity(x_, name=_TFOUTPUT)
            actual, expected = self._run(output, {}, {})
            self.assertAllClose(expected, actual)

    def test_unstack_axis(self):
        for axis in [0, 1]:
            tf.reset_default_graph()
            x_val = np.random.randn(10, 3, 4).astype("float32")
            x = tf.constant(x_val, dtype=tf.float32)
            x_ = tf.unstack(x, axis=axis)
            output = tf.identity(x_, name=_TFOUTPUT)
            actual, expected = self._run(output, {}, {})
            self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "Space2Depth not implemented, works on onnxmsrtnext")
    def test_space_to_depth(self):
        x_val = make_xval([1, 2, 2, 1])
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.space_to_depth(x, block_size=2)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(OPSET < 6, "supported since opset 6")
    def test_addn(self):
        x_val = np.arange(3*2*3).astype("float32")
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.add_n([x, x, x])
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "multiple dims not supported")
    def test_strided_slice1(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_strided_slice2(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = tf.strided_slice(x, [1, 0, 0], [2, 2, 3], [1, 1, 1])
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_strided_slice3(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[1:]
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_strided_slice4(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[:2]
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "multiple dims not supported")
    def test_strided_slice5(self):
        x_val = np.arange(3*2*3).astype("float32").reshape(3, 2, 3)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[:2, 0:1, 1:]
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "multiple dims not supported")
    def test_strided_slice6(self):
        # example from here:
        # https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/ops/strided-slice
        x_val = np.arange(5*6).astype("float32").reshape(5, 6)
        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        x_ = x[2, :]
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "fails with schema error")
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
        output = tf.identity(y, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual, rtol=1e-04)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "not correctly supported")
    def test_resize_nearest_neighbor(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        x_new_size_ = tf.constant(x_new_size)
        x_ = tf.image.resize_nearest_neighbor(x, x_new_size_)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2", "onnxmsrt"], "not correctly supported")
    def test_resize_bilinear(self):
        x_shape = [1, 15, 20, 2]
        x_new_size = [30, 40]
        x_val = np.arange(1, 1 + np.prod(x_shape)).astype("float32").reshape(x_shape)
        x = tf.placeholder(tf.float32, x_shape, name=_TFINPUT)
        x_new_size_ = tf.constant(x_new_size)
        x_ = tf.image.resize_bilinear(x, x_new_size_)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    @unittest.skip
    def test_fill(self):
        # no official fill op in onnx
        x_val = np.array([2, 3], dtype=np.int32)
        x = tf.placeholder(tf.int32, x_val.shape, name=_TFINPUT)
        x_ = tf.fill(x, 9)
        output = tf.identity(x_, name=_TFOUTPUT)
        actual, expected = self._run(output, {x: x_val}, {_INPUT: x_val})
        self.assertAllClose(expected, actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default=BACKEND,
                        choices=["caffe2", "onnxmsrt", "onnxmsrtnext", "onnx-tensorflow"],
                        help="backend to test against")
    parser.add_argument('--opset', default=OPSET,
                        help="opset to test against")
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    BACKEND = args.backend
    OPSET = args.opset
    # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
    sys.argv[1:] = args.unittest_args
    unittest.main()
