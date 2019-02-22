# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for while loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


class LoopTests(Tf2OnnxBackendTestBase):

    def test_simple_while_loop(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])

        _ = tf.identity(r, name="output")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32)}

        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_simple_while_loop_2(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        c = lambda i: tf.logical_and(tf.less(i, 10), tf.greater_equal(i, 3))
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])

        _ = tf.identity(r, name="output")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.array(3, dtype=np.int32)}

        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_while_loop_with_ta_write(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
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
        _ = tf.identity(r, name="output")
        _ = tf.identity(i_final, name="i")
        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32)}

        output_names_with_port = ["output:0", "i:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_while_loop_with_ta_read(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        inputs = tf.placeholder(tf.float32, (10,), name="input_2")
        inputs_2 = tf.identity(inputs)
        input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_2)

        inputs_3 = tf.placeholder(tf.float32, (10,), name="input_3")
        inputs_3_identity = tf.identity(inputs_3)
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
        _ = tf.identity(i_final, name="i")
        _ = tf.identity(x_final, name="x")
        _ = tf.identity(y_final, name="y")
        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32),
                     "input_3:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}

        output_names_with_port = ["i:0", "x:0", "y:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @unittest.skip("bug in onnxruntime")
    def test_while_loop_with_ta_read_reference_outer_input_directly(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        inputs = tf.placeholder(tf.float32, (10,), name="input_2")
        input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs)

        inputs_3 = tf.placeholder(tf.float32, (10,), name="input_3")
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
        _ = tf.identity(i_final, name="i")
        _ = tf.identity(x_final, name="x")
        _ = tf.identity(y_final, name="y")
        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32),
                     "input_3:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}

        output_names_with_port = ["i:0", "x:0", "y:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_while_loop_with_ta_read_and_write(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        inputs = tf.placeholder(tf.float32, (10,), name="input_2")

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
        _ = tf.identity(i_final, name="i")
        _ = tf.identity(out_final.stack(), name="output_ta")
        input_names_with_port = ["input_1:0", "input_2:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}

        output_names_with_port = ["i:0", "output_ta:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_map_fn(self):
        def fn0(elem):
            res = elem + elem * elem
            return res

        def fn1(elem):
            res = elem[0] * elem[1] + elem[0]
            return res

        x_val = 100 * np.random.random_sample([2, 10]).astype(np.float32)
        y_val = 100 * np.random.random_sample([2, 10]).astype(np.float32)

        # test fn0
        x = tf.placeholder(tf.float32, shape=x_val.shape, name="input_0")
        x_ = tf.identity(x)
        res_ = tf.map_fn(fn0, x_, dtype=tf.float32)
        _ = tf.identity(res_, name="output_0")
        feed_dict = {"input_0:0": x_val}
        input_names_with_port = ["input_0:0"]
        output_names_with_port = ["output_0:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-5)
        tf.reset_default_graph()

        # test fn1
        x = tf.placeholder(tf.float32, shape=x_val.shape, name="input_0")
        y = tf.placeholder(tf.float32, shape=y_val.shape, name="input_1")
        x_ = tf.identity(x)
        y_ = tf.identity(y)
        res_ = tf.map_fn(fn1, (x_, y_), dtype=tf.float32)
        _ = tf.identity(res_, name="output_0")
        feed_dict = {"input_0:0": x_val, "input_1:0": y_val}
        input_names_with_port = ["input_0:0", "input_1:0"]
        output_names_with_port = ["output_0:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-5)
        tf.reset_default_graph()


if __name__ == '__main__':
    unittest_main()
