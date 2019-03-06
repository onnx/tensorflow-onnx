# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for tf.cond and tf.case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test
# pylint: disable=abstract-method,arguments-differ

class CondTests(Tf2OnnxBackendTestBase):

    def test_simple_cond(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1
        res = tf.cond(x[0] < y[0], lambda: x + y, lambda: x - y, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_with_const_branch(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        true_const = tf.constant(True, name="true_const", dtype=tf.bool)

        def cond_graph():
            with tf.name_scope("cond_graph", "cond_graph", [x, y]):
                b = tf.constant(np.array([2, 1, 3], dtype=np.float32), name="b", dtype=tf.float32)
                return b

        res = tf.cond(true_const, lambda: x + y, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    @unittest.skip("a very special case that true and false branch of tf.cond only "
                   "contain a const node, which depends on Switch per control inputs")
    def test_cond_with_only_const(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")

        def cond_graph():
            with tf.name_scope("cond_graph", "cond_graph", [x, y]):
                b = tf.constant(10, name="b", dtype=tf.float32)
                return b

        res = tf.cond(x[0] < y[0], cond_graph, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_with_multi_merge(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1
        res = tf.cond(x[0] < y[0], lambda: [x, x + y], lambda: [x, x - y], name="test")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_with_replicate_output(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1
        res = tf.cond(x[0] < y[0], lambda: [x, x], lambda: [y, y], name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_nest_cond(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1

        def cond_graph():
            def cond_graph1():
                def cond_graph2():
                    return tf.cond(x[0] < y[0], lambda: x + y, lambda: tf.square(y))

                return tf.cond(tf.reduce_any(x < y), cond_graph2, cond_graph2)

            return tf.cond(x[0] > y[0], cond_graph1, cond_graph1)

        res = tf.cond(x[0] < y[0], cond_graph, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_while_loop_in_cond(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")

        def cond_graph():
            b = tf.constant(np.array([0], dtype=np.int32), dtype=tf.int32)
            # while_loop
            c = lambda y: tf.reduce_any(tf.less(y, 10))
            b = lambda i: tf.add(y, 1)
            return tf.while_loop(c, b, [y])

        res = tf.cond(x[0] < y[0], lambda: x, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_in_while_loop(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        inputs = tf.placeholder(tf.float32, (10,), name="input_2")

        inputs_2 = tf.identity(inputs)
        input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_2)
        output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        c = lambda i, *_: tf.logical_and(tf.less(i, 10), i >= 0)

        def b(i, out_ta):
            new_i = tf.add(i, 1)
            x = input_ta.read(i)
            x = tf.cond(x > 0, lambda: x - 1, lambda: x + 3)
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

    def test_simple_case(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1
        res = tf.case([(tf.reduce_all(x < 1), lambda: x + y), (tf.reduce_all(y > 0), lambda: tf.square(y))],
                      default=lambda: x, name="test_case")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_case_with_exclusive(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1
        res = tf.case([(tf.reduce_all(x < 1), lambda: x + y), (tf.reduce_all(y > 0), lambda: tf.square(y))],
                      default=lambda: x, name="test_case", exclusive=True)
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_case_without_default_branch(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1
        res = tf.case([(tf.reduce_all(x < 1), lambda: x + y), (tf.reduce_all(y > 0), lambda: tf.square(y))])
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_case_with_multi_merge(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1
        res = tf.case(
            [(tf.reduce_all(x < 1), lambda: [x + y, x - y]), (tf.reduce_all(y > 0), lambda: [tf.abs(x), tf.square(y)])],
            default=lambda: [x, y], name="test_case"
        )
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_nest_case(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        x = x + 1
        y = y + 1

        def case_graph():
            return tf.case(
                [(tf.reduce_all(x < 1), lambda: x + y), (tf.reduce_all(y > 0), lambda: tf.square(y))],
                default=lambda: x - y,
                name="test_case"
            )

        res = tf.case([(x[0] > 0, case_graph), (x[0] < 0, case_graph)], default=lambda: x - y)
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)


if __name__ == '__main__':
    unittest_main()
