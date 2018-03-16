# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import tempfile
import unittest
from collections import namedtuple

import numpy as np
import tensorflow as tf
from onnx import helper

import tf2onnx.utils
from tf2onnx.tfonnx import process_tf_graph


TMPPATH = tempfile.mkdtemp()

BACKEND = "caffe2"
BACKEND = "onnxmsrt"

NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]


# pylint: disable=C0111


def onnx_pretty(g, args=None):
    model_proto = g.make_model("converted from {}".format(args.input), args.inputs, args.outputs)
    return helper.printable_graph(model_proto.graph)


class Tf2OnnxBackendTests(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        tf.reset_default_graph()
        # reset name generation on every test
        tf2onnx.utils.INTERNAL_NAME = 1
        np.random.seed(1)  # Make it reproducible.

        arg = namedtuple("Arg", "input inputs outputs verbose continue_on_error")
        self._args0 = arg(input="test", inputs=[], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args1 = arg(input="test", inputs=["input:0"], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args2 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args3 = arg(input="test", inputs=["input1:0", "input2:0", "prob:0"], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args4 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output1:0", "output2:0"],
                          verbose=False, continue_on_error=False)

    @staticmethod
    def assertAllClose(expected, actual, **kwargs):
        np.testing.assert_allclose(expected, actual, **kwargs)

    @staticmethod
    def run_onnxcaffe2(onnx_graph, inputs):
        import onnx_caffe2.backend
        prepared_backend = onnx_caffe2.backend.prepare(onnx_graph)
        results = prepared_backend.run(inputs)
        return results

    @staticmethod
    def run_onnxnumpy(onnx_graph, inputs):
        import onnxnumpy
        g = onnxnumpy.OnnxNumpy(onnx_graph.graph)
        results = {}
        results = g.run(inputs)
        return results

    @staticmethod
    def run_onnxmsrt(onnx_graph, inputs, output_names, test_name):
        import lotus
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())

        m = lotus.ModelExecutor(model_path)
        results = m.run(output_names, inputs)
        return results[0]

    @staticmethod
    def run_onnxcntk(onnx_graph, inputs, test_name):
        import cntk as C
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())
        z = C.Function.load(model_path, format=C.ModelFormat.ONNX)
        results = np.squeeze(z.eval({z.arguments[0]: [inputs["input0:0"]]}))
        return results

    def validate_onnx(self, g, args, input_dict, expected):
        model_proto = g.make_model("test", args.inputs, args.outputs)
        if BACKEND == "onnxmsrt":
            y = self.run_onnxmsrt(model_proto, input_dict, args.outputs, self._testMethodName)
        elif BACKEND == "cntk":
            # TODO: not tested
            y = self.run_onnxcntk(model_proto, input_dict, self._testMethodName)
        elif BACKEND == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
            if y:
                y = y[0]
        elif BACKEND == "onnxnumpy":
            y = self.run_onnxnumpy(model_proto, input_dict)
            y = y[args.outputs[0]]
        else:
            raise ValueError("unknown backend")
        return y

    def _test_expand_dims(self, idx):
        tf.reset_default_graph()
        x_val = np.linspace(1, 12, 12).astype("float32").reshape(3, 4)
        x = tf.placeholder(tf.float32, shape=x_val.shape, name='input1')
        op = tf.expand_dims(x, idx)
        with tf.Session() as sess:
            output = tf.identity(op, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={x: x_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": x_val}, expected)
            self.assertAllClose(expected, actual)

    def test_expand_dims(self):
        for i in [-1, 0, 1, -2]:
            self._test_expand_dims(i)

    def test_maxppol(self):
        image = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                         dtype=np.float32).reshape(1, 4, 4, 1)
        with tf.Session() as sess:
            image_ = tf.placeholder(tf.float32, shape=image.shape, name='input1')
            mp = tf.nn.max_pool(image_, [1, 2, 2, 1], [1, 1, 1, 1], padding="VALID")
            output = tf.identity(mp, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={image_: image})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": image}, expected)
            self.assertAllClose(expected, actual)

    def test_avgppol(self):
        image = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                         dtype=np.float32).reshape(1, 4, 4, 1)

        with tf.Session() as sess:
            image_ = tf.placeholder(tf.float32, shape=image.shape, name='input1')
            mp = tf.nn.avg_pool(image_, [1, 2, 2, 1], [1, 1, 1, 1], padding="VALID")
            output = tf.identity(mp, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={image_: image})

            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": image}, expected)
            self.assertAllClose(expected, actual)

    def _conv_test(self, image, w, strides=None, padding="VALID"):
        if strides is None:
            strides = [1, 1, 1, 1]
        tf.reset_default_graph()
        kernel = tf.constant(w, dtype=tf.float32, name='k')
        with tf.Session() as sess:
            image_ = tf.placeholder(tf.float32, shape=image.shape, name='input1')
            conv = tf.nn.conv2d(image_, kernel, strides=strides, padding=padding)
            output = tf.identity(conv, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={image_: image})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": image}, expected)
        return expected, actual

    def test_conv2d_1(self):
        kernel_shape = [3, 3, 1, 1]
        image = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor as NCHW
                            [5., 6., 7., 8., 9.],
                            [10., 11., 12., 13., 14.],
                            [15., 16., 17., 18., 19.],
                            [20., 21., 22., 23., 24.]]]], dtype=np.float32).transpose(NCHW_TO_NHWC)
        # If we want a random image
        # image = np.random.uniform(1, 25, 25).astype(np.float32).reshape([1, 5, 5, 1])

        # the shape of image is [1,5,5,1] in NHWC
        # W = np.ones(kernel_shape, dtype=np.float32) # WORKS
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape(kernel_shape)  # WORKS
        # If we want a random kernel (numerical errors, ~5%)
        # W = np.random.uniform(1, 9, 9).astype(np.float32).reshape(kernel_shape)
        expected, actual = self._conv_test(image, w)
        self.assertAllClose(expected, actual)

    def test_conv2d_2(self):
        kernel_shape = [3, 3, 1, 1]
        image = np.array([[4, 3, 1, 0],
                          [2, 1, 0, 1],
                          [1, 2, 4, 1],
                          [3, 1, 0, 2]], dtype=np.float32).reshape([1, 4, 4, 1])
        w = np.array([[1, 0, 1],
                      [2, 1, 0],
                      [0, 0, 1]], dtype=np.float32).reshape(kernel_shape)
        expected, actual = self._conv_test(image, w)
        self.assertAllClose(expected, actual)

    def test_conv2d_3(self):
        image = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor as NCHW
                            [5., 6., 7., 8., 9.],
                            [10., 11., 12., 13., 14.],
                            [15., 16., 17., 18., 19.],
                            [20., 21., 22., 23., 24.]]]], dtype=np.float32).transpose(NCHW_TO_NHWC)
        w = np.array([[2., 1., 1.],
                      [1., 3., 1.],
                      [1., 1., 4.]], dtype=np.float32).reshape([3, 3, 1, 1])
        expected, actual = self._conv_test(image, w)
        self.assertAllClose(expected, actual)

    def test_conv2d_4(self):
        kernel_shape = [3, 3, 1, 1]
        image = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor as NCHW
                            [5., 6., 7., 8., 9.],
                            [10., 11., 12., 13., 14.],
                            [15., 16., 17., 18., 19.],
                            [20., 21., 22., 23., 24.]]]], dtype=np.float32).transpose(NCHW_TO_NHWC)
        w = np.random.random_sample(kernel_shape).astype(np.float32)
        expected, actual = self._conv_test(image, w, padding="SAME")
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_conv2d_5(self):
        kernel_shape = [3, 3, 1, 2]
        image = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor as NCHW
                            [5., 6., 7., 8., 9.],
                            [10., 11., 12., 13., 14.],
                            [15., 16., 17., 18., 19.],
                            [20., 21., 22., 23., 24.]]]], dtype=np.float32).transpose(NCHW_TO_NHWC)
        w = np.random.random_sample(kernel_shape).astype(np.float32)
        expected, actual = self._conv_test(image, w, padding="SAME")
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_conv2d_6(self):
        image_shape = [1, 35, 35, 288]  # out: [1, 17, 17, 384]
        kernel_shape = [3, 3, 288, 384]
        strides = [1, 2, 2, 1]
        image_val = np.arange(1, 1 + np.prod(image_shape)).astype("float32").reshape(image_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        expected, actual = self._conv_test(image_val, kernel_val, strides=strides, padding="VALID")
        self.assertAllClose(expected, actual, rtol=1e-05)

    def test_conv2d_transpose(self):
        x_shape = [2, 6, 4, 3]
        output_shape = [2, 13, 9, 2]
        kernel_shape = [3, 3, 2, 3]
        strides = [1, 2, 2, 1]
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        kernel_val = np.random.random_sample(kernel_shape).astype(np.float32)
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, shape=x_shape, name='input1')
            f = tf.constant(kernel_val, name="kernel", dtype=tf.float32)
            conv = tf.nn.conv2d_transpose(x, f, output_shape, strides=strides, padding="VALID")
            output = tf.identity(conv, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={x: x_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": x_val}, expected)
            self.assertAllClose(expected, actual, rtol=1e-05)

    def test_depthwiseconv_0(self):
        image_shape = [1, 3, 4, 3]
        kernel_shape = [3, 3, 3, 3]
        image_val = np.arange(1, 1 + np.prod(image_shape)).astype("float32").reshape(image_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        with tf.Session() as sess:
            kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
            image = tf.placeholder(tf.float32, shape=image_val.shape, name='input1')
            conv = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='VALID')
            output = tf.identity(conv, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={image: image_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": image_val}, expected)
            # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
            self.assertAllClose(expected, actual, rtol=0.08)

    def test_depthwiseconv_1(self):
        image_shape = [1, 112, 112, 32]
        kernel_shape = [3, 3, 32, 1]
        image_val = np.arange(1, 1 + np.prod(image_shape)).astype("float32").reshape(image_shape)
        kernel_val = np.arange(1, 1 + np.prod(kernel_shape)).astype("float32").reshape(kernel_shape)
        with tf.Session() as sess:
            kernel = tf.constant(kernel_val, dtype=tf.float32, name='k')
            image = tf.placeholder(tf.float32, shape=image_val.shape, name='input1')
            conv = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='VALID')
            output = tf.identity(conv, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={image: image_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": image_val}, expected)
            # rtol is a bit high, 2 values have a bit high error. Maybe use different input data.
            self.assertAllClose(expected, actual, rtol=0.08)

    @unittest.skip
    def test_lrn(self):
        # FIXME: numerical results are not correct
        image_shape = [1, 3, 4, 3]
        image_val = np.arange(1, 1 + np.prod(image_shape)).astype("float32").reshape(image_shape)
        image = tf.placeholder(tf.float32, shape=image_val.shape, name='input1')
        with tf.Session() as sess:
            op = tf.nn.local_response_normalization(image)
            output = tf.identity(op, name="output")
            sess.run(tf.global_variables_initializer())
            expected = sess.run(output, feed_dict={image: image_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": image_val}, expected)
            self.assertAllClose(expected, actual, rtol=1e-05)

    def test_abs(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.abs(x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_const(self):
        with tf.Session() as sess:
            xin = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
            x = tf.constant(xin, name="input")
            x_ = tf.identity(x)
            output = tf.add(x_, x_, name="output")
            expected = sess.run(output)
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_add(self):
        with tf.Session() as sess:
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.add(x, x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_add_bcast(self):
        with tf.Session() as sess:
            # if we'd broadcast 2,2 to 2,1 onnxmsrt will fail
            x1_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            x2_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32).reshape((2, 2, 2))
            x1 = tf.placeholder(tf.float32, x1_val.shape, name="input1")
            x2 = tf.placeholder(tf.float32, x2_val.shape, name="input2")
            x_ = tf.add(x1, x2)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x1: x1_val, x2: x2_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": x1_val, "input2:0": x2_val}, expected)
            self.assertAllClose(expected, actual)

    def test_matmul(self):
        with tf.Session() as sess:
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.matmul(x, x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_sub(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.subtract(x, x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_multiply(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.multiply(x, x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_div(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.realdiv(x, x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_exp(self):
        with tf.Session() as sess:
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.exp(x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual, rtol=1e-05)

    def test_log(self):
        with tf.Session() as sess:
            xin = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.log(x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_gather(self):
        with tf.Session() as sess:
            xin = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
            idx = np.array([1, 0, 2], dtype=np.int32)
            idx_flattened = np.array([i * xin.shape[1] + idx for i in range(0, xin.shape[0])])
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.gather(tf.reshape(x, [-1]), tf.constant(idx_flattened))
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_neg(self):
        with tf.Session() as sess:
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.negative(x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_square(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.square(x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_min(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 2], name="input1")
            x2 = tf.placeholder(tf.float32, [2, 2], name="input2")
            mi = tf.minimum(x1, x2)
            output = tf.identity(mi, name="output")
            xin1 = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
            xin2 = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x1: xin1, x2: xin2})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": xin1, "input2:0": xin2}, expected)
            self.assertAllClose(expected, actual)

    def test_logicaland(self):
        with tf.Session() as sess:
            xin1 = np.array([1, 0, 1, 1], dtype=np.bool).reshape((2, 2))
            xin2 = np.array([0, 1, 1, 1], dtype=np.bool).reshape((2, 2))
            x1 = tf.placeholder(tf.bool, [2, 2], name="input1")
            x2 = tf.placeholder(tf.bool, [2, 2], name="input2")
            mi = tf.logical_and(x1, x2)
            output = tf.identity(mi, name="output")
            expected = sess.run(output, feed_dict={x1: xin1, x2: xin2})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": xin1, "input2:0": xin2}, expected)
            self.assertAllClose(expected, actual)

    def test_greater(self):
        with tf.Session() as sess:
            xin1 = np.array([4, 2, 4, 1], dtype=np.float32).reshape((2, 2))
            xin2 = np.array([2, 4, 4, 1], dtype=np.float32).reshape((2, 2))
            x1 = tf.placeholder(tf.float32, [2, 2], name="input1")
            x2 = tf.placeholder(tf.float32, [2, 2], name="input2")
            mi = tf.greater(x1, x2)
            output = tf.identity(mi, name="output")
            expected = sess.run(output, feed_dict={x1: xin1, x2: xin2})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": xin1, "input2:0": xin2}, expected)
            self.assertAllClose(expected, actual)

    def test_sequeeze(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2, 1], name="input1")
            x_ = tf.squeeze(x)
            output = tf.identity(x_, name="output")
            xin1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2, 1))
            expected = sess.run(output, feed_dict={x: xin1})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": xin1}, expected)
            self.assertAllClose(expected, actual)

    def test_transpose(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.transpose(x)  # perm=[1,0])
            output = tf.identity(x_, name="output")
            xin1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape((2, 3))
            expected = sess.run(output, feed_dict={x: xin1})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": xin1}, expected)
            self.assertAllClose(expected, actual)

    def test_reshape(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input1")
            shape = tf.constant([1, 4])
            x_ = tf.reshape(x, shape)
            output = tf.identity(x_, name="output")
            xin1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin1})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": xin1}, expected)
            self.assertEqual(expected.shape, actual.shape)
            self.assertAllClose(expected, actual)

    def test_relu(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.nn.relu(x)
            output = tf.identity(x_, name="output")
            xin = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_elu(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.nn.elu(x)
            output = tf.identity(x_, name="output")
            xin = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_tanh(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.tanh(x)
            output = tf.identity(x_, name="output")
            xin = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual, rtol=1e-05)

    def test_relu6(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.nn.relu6(x)
            output = tf.identity(x_, name="output")
            xin = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_concat(self):
        with tf.Session() as sess:
            t1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            t2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
            t3 = np.array([[13, 14, 15], [16, 17, 18]], dtype=np.float32)
            x1 = tf.placeholder(tf.float32, t1.shape, name="input1")
            x2 = tf.placeholder(tf.float32, t2.shape, name="input2")
            x3 = tf.placeholder(tf.float32, t3.shape, name="input3")
            x_ = tf.concat([x1, x2, x3], 0)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x1: t1, x2: t2, x3: t3})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args2, {"input1:0": t1, "input2:0": t2, "input3:0": t3}, expected)
            self.assertAllClose(expected, actual)

    def test_pow(self):
        with tf.Session() as sess:
            e = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
            xin = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.pow(x, tf.constant(e))
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_embedding_lookup(self):
        with tf.Session() as sess:
            t_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
            t = tf.constant(t_val)
            x_val = np.array([[1]], dtype=np.int32)
            x = tf.placeholder(tf.int32, x_val.shape, name="input1")
            x_ = tf.nn.embedding_lookup(t, x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: x_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": x_val}, expected)
            self.assertAllClose(expected, actual)

    def test_slice(self):
        with tf.Session() as sess:
            t0 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
            t1 = tf.constant([0, 1], dtype=tf.int32)
            t2 = tf.constant([2, 2], dtype=tf.int32)
            x0 = tf.placeholder(tf.float32, t0.shape, name="input1")
            x_ = tf.slice(x0, t1, t2)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x0: t0})
            g = process_tf_graph(sess.graph)
            self.assertIsNotNone(g)
            actual = self.validate_onnx(g, self._args1, {"input1:0": t0}, expected)
            self.assertAllClose(expected, actual)

    def test_split(self):
        with tf.Session() as sess:
            t0_shape = [5, 30]
            t0 = np.linspace(1.0, 5 * 30.0, 5 * 30).astype(np.float32).reshape(t0_shape)
            x0 = tf.placeholder(tf.float32, t0.shape, name="input1")
            x_, _, _ = tf.split(x0, [4, 15, 11], 1)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x0: t0})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": t0}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_reducesum(self):
        # not supported by onnx-caffe2
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input1")
            x_ = tf.reduce_sum(x)
            output = tf.identity(x_, name="output")
            xin1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin1})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": xin1}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_sqrt(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.sqrt(x)
            output = tf.identity(x_, name="output")
            xin = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_rsqrt(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.rsqrt(x)
            output = tf.identity(x_, name="output")
            xin = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual, rtol=1e-05)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_reciprocal(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.reciprocal(x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual, rtol=1e-04)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_reducemax(self):
        # not supported by onnx-caffe2
        with tf.Session() as sess:
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.reduce_max(x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual, rtol=1e-05)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_reduceprod(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.reduce_prod(x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND == "caffe2", "not supported in caffe2")
    def test_reducemean(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.reduce_mean(x)
            output = tf.identity(x_, name="output")
            xin = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skip
    def test_slice1(self):
        # FIXME: caffe2 complaining about doing only 1 dimension
        with tf.Session() as sess:
            t0 = np.array([[[1, 1, 1], [2, 2, 2]],
                           [[3, 3, 3], [4, 4, 4]],
                           [[5, 5, 5], [6, 6, 6]]], dtype=np.float32)
            t1 = tf.constant([1, 0, 0], dtype=tf.int32)
            t2 = tf.constant([1, 1, 3], dtype=tf.int32)
            x0 = tf.placeholder(tf.float32, t0.shape, name="input1")
            x_ = tf.slice(x0, t1, t2)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x0: t0})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input1:0": t0}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND in ["caffe2"], "issue with broadcastnig scalar")
    def test_pow_scalar(self):
        with tf.Session() as sess:
            e = np.array(2.0, dtype=np.float32)
            xin = np.array([4.0, 16.0, 4.0, 1.6], dtype=np.float32)
            x = tf.placeholder(tf.float32, xin.shape, name="input")
            x_ = tf.pow(x, tf.constant(e))
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skipIf(BACKEND == "caffe2", "not supported correctly in caffe2")
    def test_pad(self):
        with tf.Session() as sess:
            x_val = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], dtype=np.float32)
            x = tf.placeholder(tf.float32, x_val.shape, name="input")
            paddings = tf.constant([[0, 0, ], [2, 0]])
            op = tf.pad(x, paddings, "CONSTANT")
            output = tf.identity(op, name="output")
            expected = sess.run(output, feed_dict={x: x_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": x_val}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skip
    def test_randomuniform(self):
        # not supported by onnxmsrt or caffe2
        with tf.Session() as sess:
            shape = tf.constant([2, 3], name="shape")
            x_ = tf.random_uniform(shape, name="rand", dtype=tf.float32)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            output = tf.identity(x_, name="output")
            expected = sess.run(output)
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args4, {}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skip
    def test_argminmax(self):
        # TODO: fails on onnxmsrt caffe2
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.argmin(x, axis=0)
            output = tf.identity(x_, name="output")
            xin = np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertIsNotNone(g)
            self.assertAllClose(expected, actual)

    def test_cast(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 2], name="input")
            x_ = tf.cast(x, tf.int32)
            output = tf.identity(x_, name="output")
            x_val = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32).reshape((2, 2))
            expected = sess.run(output, feed_dict={x: x_val})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": x_val}, expected)
            self.assertAllClose(expected, actual)

    @unittest.skip
    def test_onehot(self):
        # FIXME via onnx-ml ?
        with tf.Session() as sess:
            xin = np.array([0, 1, 2], dtype=np.int32)
            depth = 3
            x = tf.placeholder(tf.int32, xin.shape, name="input")
            x_ = tf.one_hot(x, depth)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)

    def test_flatten0(self):
        with tf.Session() as sess:
            xin = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
            x = tf.placeholder(tf.float32, [None, 3, 3], name="input")
            x_ = tf.layers.flatten(x)
            output = tf.identity(x_, name="output")
            expected = sess.run(output, feed_dict={x: xin})
            g = process_tf_graph(sess.graph)
            actual = self.validate_onnx(g, self._args1, {"input:0": xin}, expected)
            self.assertAllClose(expected, actual)


if __name__ == "__main__":
    unittest.main()
