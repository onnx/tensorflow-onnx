# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for grublock."""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, check_gru_count, check_tf_max_version
from common import check_opset_after_tf_version
from tf2onnx.tf_loader import is_tf2


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test
# pylint: disable=invalid-name

if is_tf2():
    MultiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell
    dynamic_rnn = tf.compat.v1.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.compat.v1.nn.bidirectional_dynamic_rnn
else:
    GRUBlockCell = tf.contrib.rnn.GRUBlockCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell
    dynamic_rnn = tf.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn

# pylint: enable=invalid-name


# TODO: as a workaround, set batch_size to 1 for now to bypass a onnxruntime bug, revert it when the bug is fixed
class GRUBlockTests(Tf2OnnxBackendTestBase):

    def run_test_case(self, *args, **kwargs):  #pylint: disable=arguments-differ
        # TF GRU has an unknown dim
        tmp = self.config.allow_missing_shapes
        self.config.allow_missing_shapes = True
        try:
            super().run_test_case(*args, **kwargs)
        finally:
            self.config.allow_missing_shapes = tmp

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = GRUBlockCell(
                units)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_multiple_dynamic_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            gru_output_list = []
            gru_cell_state_list = []
            # no scope
            cell = GRUBlockCell(units)
            outputs, cell_state = dynamic_rnn(cell, x, dtype=tf.float32)
            gru_output_list.append(outputs)
            gru_cell_state_list.append(cell_state)

            # given scope
            cell = GRUBlockCell(units)
            with variable_scope.variable_scope("root1") as scope:
                outputs, cell_state = dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=[4], scope=scope)
            gru_output_list.append(outputs)
            gru_cell_state_list.append(cell_state)

            return tf.identity(gru_output_list, name="output"), tf.identity(gru_cell_state_list, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 2))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_gru_seq_length_is_const(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = GRUBlockCell(
                units)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32,
                sequence_length=[5])
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_gru_seq_length_is_not_const(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        y_val = np.array([5], dtype=np.int32)

        def func(x, seq_length):
        # no scope
            cell = GRUBlockCell(
                units)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32,
                sequence_length=tf.identity(seq_length))

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_gru_placeholder_input(self):
        units = 5
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * 1)
        def func(x):
            # no scope
            cell = GRUBlockCell(
                units)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)  # by default zero initializer is used

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_gru_ch_zero_state_initializer(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = GRUBlockCell(
                units)

            # defining initial state
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                initial_state=initial_state,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_gru_random_weights(self):
        hidden_size = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # no scope
            cell = GRUBlockCell(
                hidden_size)

            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.0001,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_gru_random_weights2(self):
        hidden_size = 128
        batch_size = 1
        x_val = np.random.randn(1, 133).astype('f')
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = GRUBlockCell(
                hidden_size)

            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.01,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_gru_output_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            cell1 = GRUBlockCell(units)
            outputs, _ = dynamic_rnn(cell1, x, dtype=tf.float32)
            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.0001,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_gru_state_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            cell1 = GRUBlockCell(units)
            _, cell_state = dynamic_rnn(cell1, x, dtype=tf.float32)
            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.0001,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_bigru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # bigru, no scope
            cell1 = GRUBlockCell(
                units)
            cell2 = GRUBlockCell(
                units)
            outputs, cell_state = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_bigru_output_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # bigru, no scope
            cell1 = GRUBlockCell(
                units)
            cell2 = GRUBlockCell(
                units)
            outputs, _ = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_bigru_state_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # bigru, no scope
            cell1 = GRUBlockCell(
                units)
            cell2 = GRUBlockCell(
                units)
            _, cell_state = bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)

            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_bidirectional_but_one_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # bigru, no scope
            cell = GRUBlockCell(
                units)
            outputs, cell_state = bidirectional_dynamic_rnn(
                cell,
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_bidirectional_but_one_gru_and_output_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # bigru, no scope
            cell = GRUBlockCell(
                units)
            outputs, _ = bidirectional_dynamic_rnn(
                cell,
                cell,
                x,
                dtype=tf.float32)
            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-07,
                           graph_validator=lambda g: check_gru_count(g, 1))

    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_bidirectional_but_one_gru_and_state_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # bigru, no scope
            cell = GRUBlockCell(
                units)
            _, cell_state = bidirectional_dynamic_rnn(
                cell,
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))


if __name__ == '__main__':
    unittest_main()
