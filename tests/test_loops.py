# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for while loops."""

import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, check_tf_min_version, \
    check_onnxruntime_min_version, check_tfjs_max_version, skip_tflite
from tf2onnx.tf_loader import is_tf2


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

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

class LoopTests(Tf2OnnxBackendTestBase):

    def test_simple_while_loop(self):
        def func(i):
            # tf2 works a little different than tf1 - that is why there is a is_tf2() here
            if is_tf2():
                one = tf.constant(np.array([1], dtype=np.int32))
            else:
                one = tf.constant(np.array(1, dtype=np.int32))
            c = lambda i: tf.less(i, 10)
            b = lambda i: tf.add(i, one)
            r = tf.while_loop(c, b, [i])
            if is_tf2():
                r = tf.reshape(r, [-1])
            return tf.identity(r, name=_TFOUTPUT)

        if is_tf2():
            x_val = np.array([0], dtype=np.int32)
        else:
            x_val = np.array(0, dtype=np.int32)
        self.run_test_case(func, {_INPUT: x_val}, [], [_OUTPUT], rtol=1e-06)

    def test_simple_while_loop_2(self):
        def func(i):
            # tf2 works a little different than tf1 - that is why there is a is_tf2() here
            if is_tf2():
                one = tf.constant(np.array([1], dtype=np.int32))
            else:
                one = tf.constant(np.array(1, dtype=np.int32))
            c = lambda i: tf.logical_and(tf.less(i, 10), tf.greater_equal(i, 3))
            b = lambda i: tf.add(i, one)
            r = tf.while_loop(c, b, [i])
            if is_tf2():
                r = tf.reshape(r, [-1])
            return tf.identity(r, name="output")
        if is_tf2():
            x_val = np.array([3], dtype=np.int32)
        else:
            x_val = np.array(3, dtype=np.int32)
        self.run_test_case(func, {_INPUT: x_val}, [], [_OUTPUT], rtol=1e-06)

    @check_tfjs_max_version("3.15", "failed when tfjs version > 3.15")
    def test_while_loop_with_ta_write(self):
        def func(i):
            output_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

            # todo: we cannot support i >= 3 for now, because in this case, TensorArray by default will
            # leave 0 for in the first 3 index.
            c = lambda i, *_: tf.logical_and(tf.less(i, 10), i >= 0)

            def b(i, out_ta):
                new_i = tf.add(i, 1)
                out_ta_new = out_ta.write(i, i)
                return new_i, out_ta_new

            i_final, ta_final = tf.while_loop(c, b, [i, output_ta])
            r = ta_final.stack()
            return tf.identity(r, name="output"), tf.identity(i_final, name="i")

        x_val = np.array(0, dtype=np.int32)
        output_names_with_port = ["output:0", "i:0"]
        self.run_test_case(func, {_INPUT: x_val}, [], output_names_with_port, rtol=1e-06)

    def test_while_loop_with_ta_read_simple(self):
        def func(i, inputs_2):
            input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_2)
            c = lambda i, *_: tf.less(i, 10)
            res = tf.constant(0.)
            def b(i, r):
                new_i = tf.add(i, 1)
                x = input_ta.read(i)
                r = tf.add(r, x)
                return new_i, r

            i_final, x_final = tf.while_loop(c, b, [i, res])
            return tf.identity(i_final, name="i"), tf.identity(x_final, name="x")
        input_names_with_port = ["input_1:0", "input_2:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], dtype=np.float32)}
        output_names_with_port = ["i:0", "x:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_while_loop_with_ta_read(self):
        def func(i, input_2, input_3):
            inputs_2 = tf.identity(input_2)
            input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_2)

            inputs_3_identity = tf.identity(input_3)
            input_ta_3 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_3_identity)

            c = lambda i, *_: tf.logical_and(tf.less(i, 10), i >= 0)
            res = tf.constant(0.)
            res2 = tf.constant(1.)

            def b(i, res, res2):
                new_i = tf.add(i, 1)
                x = input_ta.read(i)
                x = x + 3
                y = input_ta_3.read(i) + res2
                return new_i, x, y

            i_final, x_final, y_final = tf.while_loop(c, b, [i, res, res2])
            return tf.identity(i_final, name="i"), tf.identity(x_final, name="x"), tf.identity(y_final, name="y")
        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32),
                     "input_3:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}
        output_names_with_port = ["i:0", "x:0", "y:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_while_loop_with_ta_read_reference_outer_input_directly(self):
        def func(i, inputs_1, inputs_3):
            input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_1)
            input_ta_3 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_3)

            c = lambda i, *_: tf.logical_and(tf.less(i, 10), i >= 0)
            res = tf.constant(0.)
            res2 = tf.constant(1.)

            def b(i, res, res2):
                new_i = tf.add(i, 1)
                x = input_ta.read(i)
                x = x + 3
                y = input_ta_3.read(i) + res2
                return new_i, x, y

            i_final, x_final, y_final = tf.while_loop(c, b, [i, res, res2])
            return tf.identity(i_final, name="i"), tf.identity(x_final, name="x"), tf.identity(y_final, name="y")
        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32),
                     "input_3:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}
        output_names_with_port = ["i:0", "x:0", "y:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_tfjs_max_version("3.15", "failed when tfjs version > 3.15")
    def test_while_loop_with_ta_read_and_write(self):
        def func(i, inputs):
            inputs_2 = tf.identity(inputs)
            input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_2)
            output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            c = lambda i, *_: tf.logical_and(tf.less(i, 10), i >= 0)

            def b(i, out_ta):
                new_i = tf.add(i, 1)
                x = input_ta.read(i)
                x = x + 3
                out_ta_new = out_ta.write(i, x)
                return new_i, out_ta_new

            i_final, out_final = tf.while_loop(c, b, [i, output_ta])
            return tf.identity(i_final, name="i"), tf.identity(out_final.stack(), name="output_ta")

        input_names_with_port = ["input_1:0", "input_2:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}
        output_names_with_port = ["i:0", "output_ta:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_tfjs_max_version("3.15", "failed when tfjs version > 3.15")
    def test_while_loop_with_multi_scan_outputs(self):
        def func(i, inputs1, inputs2):
            inputs1_ = tf.identity(inputs1)
            inputs2_ = tf.identity(inputs2)
            input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs1_)
            input_ta2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs2_)
            output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            output_ta2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            c = lambda i, *_: tf.logical_and(tf.less(i, 10), i >= 0)

            def b(i, out_ta, out_ta2):
                new_i = tf.add(i, 1)
                x = input_ta.read(i)
                y = input_ta2.read(i)
                z = x + 3 + y
                p = x * y * 2
                out_ta_new = out_ta.write(i, z)
                out_ta_new2 = out_ta2.write(i, p)
                return new_i, out_ta_new, out_ta_new2

            i_final, out_final, out_final2 = tf.while_loop(c, b, [i, output_ta, output_ta2])
            i_final_ = tf.identity(i_final, name="i")
            out_final_ = tf.identity(out_final.stack(), name="output_ta")
            out_final2_ = tf.identity(out_final2.stack(), name="output_ta2")
            return i_final_, out_final_, out_final2_

        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32),
                     "input_3:0": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 16.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}
        output_names_with_port = ["i:0", "output_ta:0", "output_ta2:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_tfjs_max_version("3.15", "failed when tfjs version > 3.15")
    @check_onnxruntime_min_version(
        "0.5.0",
        "disable this case due to onnxruntime loop issue: https://github.com/microsoft/onnxruntime/issues/1272"
    )
    def test_while_loop_with_cond_init_false(self):
        def func(i, inputs):
            inputs_2 = tf.identity(inputs)
            input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_2)
            output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            c = lambda i, *_: tf.logical_and(i < 10, i >= 0)

            def b(i, out_ta):
                new_i = tf.add(i, 1)
                x = input_ta.read(i)
                y = x + 3
                out_ta_new = out_ta.write(i, y)
                return new_i, out_ta_new

            i_final, out_final = tf.while_loop(c, b, [i, output_ta])
            return tf.identity(i_final, name="i"), tf.identity(out_final.stack(), name="output_ta")

        input_names_with_port = ["input_1:0", "input_2:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}
        output_names_with_port = ["i:0", "output_ta:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_map_fn_1(self):
        x_val = 100 * np.random.random_sample([2, 10]).astype(np.float32)

        def fn(elem):
            res = elem + elem * elem
            return res

        def func(x):
            x = tf.identity(x)
            res_ = tf.map_fn(fn, x, dtype=tf.float32)
            return tf.identity(res_, name="output_0")

        feed_dict = {"input_0:0": x_val}
        input_names_with_port = ["input_0:0"]
        output_names_with_port = ["output_0:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-5)

    def test_map_fn_2(self):
        x_val = 100 * np.random.random_sample([2, 10]).astype(np.float32)
        y_val = 100 * np.random.random_sample([2, 10]).astype(np.float32)

        def fn(elem):
            res = elem[0] * elem[1] + elem[0]
            return res

        def func(x, y):
            x_ = tf.identity(x)
            y_ = tf.identity(y)
            res_ = tf.map_fn(fn, (x_, y_), dtype=tf.float32)
            return tf.identity(res_, name="output_0")
        feed_dict = {"input_0:0": x_val, "input_1:0": y_val}
        input_names_with_port = ["input_0:0", "input_1:0"]
        output_names_with_port = ["output_0:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-5)

    @check_tf_min_version("1.9")
    @skip_tflite("infinite loop with tflite")
    def test_simple_while_loop_var_shape(self):
        # test for while_loop with variant shape variables
        def func(i):
            const = tf.constant(np.array([2], dtype=np.int32))
            c = lambda i: tf.reduce_all(tf.shape(i) < 10)
            b = lambda i: [tf.concat([i, const], 0)]
            r = tf.while_loop(c, b, [i], shape_invariants=[tf.TensorShape([None])])
            return tf.identity(r, name="output")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.array([0], dtype=np.int32)}
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_tf_min_version("2")
    @skip_tflite("cond_graph conversion fails with tflite")
    def test_while_loop_cond_subgraphs(self):
        # test for while_loop with subgraphs in cond
        # Note: this is not working on tf1
        def func(x):
            x_dim = tf.shape(x)[0]
            r = tf.cast(tf.zeros(1), x.dtype)
            for i in tf.range(10):
                if i == x_dim:
                    break
                r += x[i]
            return tf.identity(r, name="output")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.arange(0, 15, dtype=np.int32)}
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

if __name__ == '__main__':
    unittest_main()
