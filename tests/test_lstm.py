# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for lstm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, check_lstm_count


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


class LSTMTests(Tf2OnnxBackendTestBase):
    def test_test_single_dynamic_lstm_state_is_tuple(self):
        self.internal_test_single_dynamic_lstm(True)

    def test_test_single_dynamic_lstm_state_is_not_tuple(self):
        self.internal_test_single_dynamic_lstm(False)

    def internal_test_single_dynamic_lstm(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_single_dynamic_lstm_time_major(self):
        units = 5
        seq_len = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * seq_len)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            time_major=True,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_single_dynamic_lstm_forget_bias(self):
        units = 5
        seq_len = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * seq_len)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            forget_bias=0.5)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            time_major=True,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_single_dynamic_lstm_seq_length_is_const(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        state_is_tuple = True
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32,
            sequence_length=[4, 3, 4, 5, 2, 1])

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_single_dynamic_lstm_seq_length_is_not_const(self):
        for np_dtype, tf_dtype in [[np.int32, tf.int32], [np.int64, tf.int64], [np.float32, tf.float32]]:
            tf.reset_default_graph()
            units = 5
            batch_size = 6
            x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
            x_val = np.stack([x_val] * batch_size)
            state_is_tuple = True
            x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
            initializer = init_ops.constant_initializer(0.5)

            y_val = np.array([4, 3, 4, 5, 2, 1], dtype=np_dtype)
            seq_length = tf.placeholder(tf_dtype, y_val.shape, name="input_2")

            # no scope
            cell = rnn.LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=state_is_tuple)
            outputs, cell_state = tf.nn.dynamic_rnn(
                cell,
                x,
                dtype=tf.float32,
                sequence_length=tf.identity(seq_length))

            _ = tf.identity(outputs, name="output")
            _ = tf.identity(cell_state, name="cell_state")

            feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
            input_names_with_port = ["input_1:0", "input_2:0"]
            output_names_with_port = ["output:0", "cell_state:0"]
            self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                               graph_validator=lambda g: check_lstm_count(g, 1))

    def test_single_dynamic_lstm_placeholder_input(self):
        units = 5
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * 6)
        state_is_tuple = True
        x = tf.placeholder(tf.float32, shape=(None, 4, 2), name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)  # by default zero initializer is used

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_single_dynamic_lstm_ch_zero_state_initializer(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        state_is_tuple = True
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)

        # defining initial state
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            initial_state=initial_state,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_single_dynamic_lstm_consume_one_of_ch_tuple(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)
        state_is_tuple = True
        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state.c, name="cell_state_c")
        _ = tf.identity(cell_state.h, name="cell_state_h")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state_c:0", "cell_state_h:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @unittest.skip("FIXME: disable for now for accuracy problem")
    def test_single_dynamic_lstm_random_weights(self, state_is_tuple=True):
        hidden_size = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = tf.random_uniform_initializer(-1.0, 1.0)

        # no scope
        cell = rnn.LSTMCell(
            hidden_size,
            initializer=initializer,
            state_is_tuple=state_is_tuple)

        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @unittest.skip("FIXME: disable for now for accuracy problem")
    def test_single_dynamic_lstm_random_weights2(self, state_is_tuple=True):
        hidden_size = 128
        batch_size = 1
        x_val = np.random.randn(1, 133).astype('f')
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = tf.random_uniform_initializer(0.0, 1.0)
        # no scope
        cell = rnn.LSTMCell(
            hidden_size,
            initializer=initializer,
            state_is_tuple=state_is_tuple)

        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=0.01,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_multiple_dynamic_lstm_state_is_tuple(self):
        self.internal_test_multiple_dynamic_lstm_with_parameters(True)

    def test_multiple_dynamic_lstm_state_is_not_tuple(self):
        self.internal_test_multiple_dynamic_lstm_with_parameters(False)

    def internal_test_multiple_dynamic_lstm_with_parameters(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        _ = tf.placeholder(tf.float32, x_val.shape, name="input_2")
        initializer = init_ops.constant_initializer(0.5)

        lstm_output_list = []
        lstm_cell_state_list = []
        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)
        lstm_output_list.append(outputs)
        lstm_cell_state_list.append(cell_state)

        # given scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        with variable_scope.variable_scope("root1") as scope:
            outputs, cell_state = tf.nn.dynamic_rnn(
                cell,
                x,
                dtype=tf.float32,
                sequence_length=[4, 4, 4, 4, 4, 4],
                scope=scope)
        lstm_output_list.append(outputs)
        lstm_cell_state_list.append(cell_state)

        _ = tf.identity(lstm_output_list, name="output")
        _ = tf.identity(lstm_cell_state_list, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 2))

    def test_dynamic_basiclstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        cell1 = rnn.BasicLSTMCell(
            units,
            state_is_tuple=True)

        outputs, cell_state = tf.nn.dynamic_rnn(
            cell1,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_dynamic_lstm_output_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        cell1 = rnn.LSTMCell(
            units,
            state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(
            cell1,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001, atol=1e-07,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_dynamic_lstm_state_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        cell1 = rnn.LSTMCell(
            units,
            state_is_tuple=True)

        _, cell_state = tf.nn.dynamic_rnn(
            cell1,
            x,
            dtype=tf.float32)

        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_dynamic_bilstm_state_is_tuple(self):
        self.internal_test_dynamic_bilstm_with_parameters(True)

    def test_dynamic_bilstm_state_is_not_tuple(self):
        self.internal_test_dynamic_bilstm_with_parameters(False)

    def internal_test_dynamic_bilstm_with_parameters(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bilstm, no scope
        cell1 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
        cell2 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, cell_state = tf.nn.bidirectional_dynamic_rnn(
            cell1,
            cell2,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_dynamic_bilstm_output_consumed_only(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bilstm, no scope
        cell1 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
        cell2 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell1,
            cell2,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    def test_dynamic_bilstm_state_consumed_only(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bilstm, no scope
        cell1 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
        cell2 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        _, cell_state = tf.nn.bidirectional_dynamic_rnn(
            cell1,
            cell2,
            x,
            dtype=tf.float32)

        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))


if __name__ == '__main__':
    unittest_main()
