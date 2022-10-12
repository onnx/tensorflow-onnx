# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for tf.cond and tf.case."""

import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
from common import *  # pylint: disable=wildcard-import, unused-wildcard-import


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test
# pylint: disable=abstract-method,arguments-differ

class CondTests(Tf2OnnxBackendTestBase):

    def test_simple_cond(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = x + 1
            y = y + 1
            res = tf.cond(x[0] < y[0], lambda: x + y, lambda: x - y, name="test_cond")
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_with_const_branch(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            true_const = tf.constant(True, name="true_const", dtype=tf.bool)

            def cond_graph():
                return tf.constant(np.array([2, 1, 3], dtype=np.float32), name="b", dtype=tf.float32)

            res = tf.cond(true_const, lambda: x + y, cond_graph, name="test_cond")
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    @allow_missing_shapes("TF1 has unknown dim in if output")
    def test_cond_with_multi_merge(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = x + 1
            y = y + 1
            res = tf.cond(x[0] < y[0], lambda: [x, x + y], lambda: [x, x - y], name="test")
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    @allow_missing_shapes("TF1 has unknown dim in if output")
    def test_cond_with_replicate_output(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = x + 1
            y = y + 1
            res = tf.cond(x[0] < y[0], lambda: [x, y], lambda: [y, x], name="test_cond")
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    def test_nest_cond(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = x + 1
            y = y + 1
            def cond_graph():
                def cond_graph1():
                    def cond_graph2():
                        return tf.cond(x[0] < y[0], lambda: x + y, lambda: tf.square(y))
                    return tf.cond(tf.reduce_any(x < y), cond_graph2, cond_graph2)
                return tf.cond(x[0] > y[0], cond_graph1, cond_graph1)

            res = tf.cond(x[0] < y[0], cond_graph, cond_graph, name="test_cond")
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    def test_while_loop_in_cond(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            def true_fn():
                return [x]
            def false_fn():
                # b = tf.constant(np.array([0], dtype=np.int32), dtype=tf.int32)
                # while_loop
                c = lambda y: tf.reduce_any(tf.less(y, 10))
                b = lambda i: tf.add(y, 1)
                return tf.while_loop(c, b, [y])

            res = tf.cond(x[0] < y[0], true_fn, false_fn, name="test_cond")
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    @check_tfjs_max_version("3.15", "failed when tfjs version > 3.15")
    def test_cond_in_while_loop(self):
        def func(i, inputs):
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
            return tf.identity(i_final, name="i"), tf.identity(out_final.stack(), name="output_ta")

        input_names_with_port = ["input_1:0", "input_2:0"]
        feed_dict = {"input_1:0": np.array(0, dtype=np.int32),
                     "input_2:0": np.array([2.0, 16.0, 5.0, 1.6, 5.0, 6.0, 7.0, 8.0, 9.0, 10.], dtype=np.float32)}

        output_names_with_port = ["i:0", "output_ta:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    def test_simple_case(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = tf.add(x, 1, name="add_x")
            y = tf.add(y, 1, name="add_y")
            res = tf.case([(tf.reduce_all(x < 1, name="red1"), lambda: x + y),
                           (tf.reduce_all(y > 0, name="red2"), lambda: tf.square(y))],
                          default=lambda: x, name="test_case")
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    @skip_tfjs("TFJS issue, cannot read properties of undefined (reading 'name')")
    def test_case_with_exclusive(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = x + 1
            y = y + 1
            res = tf.case([(tf.reduce_all(x < 1), lambda: x + y), (tf.reduce_all(y > 0), lambda: tf.square(y))],
                          default=lambda: x, name="test_case", exclusive=True)
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    @skip_tfjs("TFJS issue, cannot read properties of undefined (reading 'name')")
    def test_case_without_default_branch(self):
        def func(x, y):
            x = tf.add(x, 1, name="add_x")
            y = tf.add(y, 1, name="add_y")
            res = tf.case([(tf.reduce_all(x < 1), lambda: x + y),
                           (tf.reduce_all(y > 0), lambda: tf.square(y))])
            return tf.identity(res, name="output")

        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    @allow_missing_shapes("TF1 has unknown dim in if output")
    @skip_tfjs("TFJS executes model incorrectly")
    def test_case_with_multi_merge(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = x + 1
            y = y + 1
            res = tf.case(
                [(tf.reduce_all(x < 1), lambda: [x + y, x - y]),
                 (tf.reduce_all(y > 0), lambda: [tf.abs(x), tf.square(y)])],
                default=lambda: [x, y], name="test_case"
            )
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    def test_nest_case(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        def func(x, y):
            x = x + 1
            y = y + 1
            def case_graph():
                return tf.case(
                    [(tf.reduce_all(x < 1), lambda: x + y), (tf.reduce_all(y > 0), lambda: tf.square(y))],
                    default=lambda: x - y,
                    name="test_case")
            res = tf.case([(x[0] > 0, case_graph), (x[0] < 0, case_graph)], default=lambda: x - y)
            return tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port)

    @check_tf_min_version("1.8", "shape inference for Reshape op screws up")
    @check_opset_min_version(9, "ConstantOfShape")
    @allow_missing_shapes("ONNX shape inference still determines if/else shape for unknown reason")
    def test_cond_with_different_output_shape(self):
        input_shape = (10, 5, 20)
        def func(inputs, shape):
            # cheat onnx shape inference
            inputs = tf.reshape(inputs, shape)

            def pad_tensor(t, length):
                """Pads the input tensor with 0s along the first dimension up to the length.

                Args:
                  t: the input tensor, assuming the rank is at least 1.
                  length: a tensor of shape [1]  or an integer, indicating the first dimension
                    of the input tensor t after padding, assuming length <= t.shape[0].

                Returns:
                  padded_t: the padded tensor, whose first dimension is length. If the length
                    is an integer, the first dimension of padded_t is set to length
                    statically.
                """
                t_rank = tf.rank(t)
                t_shape = tf.shape(t)
                t_d0 = t_shape[0]
                pad_d0 = tf.expand_dims(length - t_d0, 0)
                pad_shape = tf.cond(
                    # shape is [3], depending on input shape
                    tf.greater(t_rank, 1), lambda: tf.concat([pad_d0, t_shape[1:]], 0),
                    # shape is always [1]
                    lambda: tf.expand_dims(length - t_d0, 0))
                padded_t = tf.concat([t, tf.zeros(pad_shape, dtype=t.dtype)], 0)
                return padded_t

            output = pad_tensor(inputs, 20)
            return tf.identity(output, name="output")
        input_names_with_port = ["input:0", "shape:0"]
        feed_dict = {
            "input:0": np.ones(input_shape, dtype=np.float32),
            "shape:0": np.array(input_shape, dtype=np.int32)
        }
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)


if __name__ == '__main__':
    unittest_main()
