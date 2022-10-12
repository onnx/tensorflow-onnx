# SPDX-License-Identifier: Apache-2.0


"""Unit tests using onnx constant folding rewriters."""

import unittest

import tensorflow as tf
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring,invalid-name,unused-argument

# pylint: disable=C0111
class ConstantFoldingTests(Tf2OnnxBackendTestBase):
    def _run_test_case(self, func, output_names_with_port, feed_dict, **kwargs):
        kwargs["convert_var_to_const"] = False
        return self.run_test_case(func, feed_dict, [], output_names_with_port, **kwargs)

    def test_concat(self):
        def func():
            t1 = [[1, 2, 3], [4, 5, 6]]
            t2 = [[7, 8, 9], [10, 11, 12]]
            x_ = tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            y_ = tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
            z_ = tf.concat([t1, t2], -1)
            return tf.identity(x_, name="output_0"), tf.identity(y_, name="output_1"), tf.identity(z_, name="output_2")
        self._run_test_case(func, ["output_0:0", "output_1:0", "output_2:0"], {})

    def test_range(self):
        def func0():
            start = tf.constant(3, dtype=tf.float32, name='start')
            limit = tf.constant(18, dtype=tf.float32, name='limit')
            delta = tf.constant(3, dtype=tf.float32, name='delta')
            x_ = tf.range(start, limit, delta)
            return tf.identity(x_, name="output_0")
        self._run_test_case(func0, ["output_0:0"], {})

        def func1():
            start = tf.constant(3, dtype=tf.float32, name='start')
            limit = tf.constant(1, dtype=tf.float32, name='limit')
            delta = tf.constant(-0.5, dtype=tf.float32, name='delta')
            x_ = tf.range(start, limit, delta)
            return tf.identity(x_, name="output_0")
        self._run_test_case(func1, ["output_0:0"], {})

        def func2():
            limit = tf.constant(5, dtype=tf.float32, name='limit')
            x_ = tf.range(limit)
            return tf.identity(x_, name="output_0")
        self._run_test_case(func2, ["output_0:0"], {})

    @unittest.skip("tensorflow op ListDiff is not supported")
    def test_bahdanau_attention_memory_layer_tensordot(self):
        def func():
            rank = tf.constant(3, dtype=tf.int32, name='rank')
            start = tf.constant(0, dtype=tf.int32, name='start')
            delta = tf.constant(1, dtype=tf.int32, name='delta')
            tensor_dot_range = tf.range(start, rank, delta)

            axes = tf.constant([2], dtype=tf.int32, name='axes')
            ge_y = tf.constant(0, dtype=tf.int32, name='ge_y')
            ge = tf.greater_equal(axes, ge_y)
            cast = tf.cast(ge, tf.int32)
            mul_1 = tf.multiply(cast, axes)

            less_y = tf.constant(0, dtype=tf.int32)
            less = tf.less(axes, less_y)
            cast_2 = tf.cast(less, tf.int32)
            add_1 = tf.add(axes, rank)
            mul_2 = tf.multiply(cast_2, add_1)

            add_2 = tf.add(mul_1, mul_2)
            out, _ = tf.setdiff1d(tensor_dot_range, add_2)
            return tf.identity(out, name="output_0")
        self._run_test_case(func, ["output_0:0"], {})


if __name__ == '__main__':
    unittest_main()
