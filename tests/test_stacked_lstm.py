# SPDX-License-Identifier: Apache-2.0

"""Unit Tests for layered lstm"""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import init_ops
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, check_lstm_count, check_opset_after_tf_version, skip_tf2

from tf2onnx.tf_loader import is_tf2


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,cell-var-from-loop
# pylint: disable=invalid-name

if is_tf2():
    LSTMCell = tf.compat.v1.nn.rnn_cell.LSTMCell
    MultiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell
    dynamic_rnn = tf.compat.v1.nn.dynamic_rnn
else:
    LSTMCell = tf.contrib.rnn.LSTMCell
    LSTMBlockCell = tf.contrib.rnn.LSTMBlockCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell
    dynamic_rnn = tf.nn.dynamic_rnn


class LSTMLayeredTests(Tf2OnnxBackendTestBase):
    @check_opset_after_tf_version("1.15", 8, "might need Scan")
    @skip_tf2()
    def test_layered_lstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            initializer = init_ops.constant_initializer(0.5)
            num_layers = 2

            # no scope
            def lstm_cell():
                return LSTMCell(
                    units,
                    initializer=initializer,
                    state_is_tuple=True)

            stacked_lstm = MultiRNNCell(
                [lstm_cell() for _ in range(num_layers)])
            outputs, cell_state = dynamic_rnn(
                stacked_lstm,
                x,
                dtype=tf.float32)
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 2))


if __name__ == '__main__':
    unittest_main()
