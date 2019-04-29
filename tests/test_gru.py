# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for gru."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, check_gru_count


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


# TODO: as a workaround, set batch_size to 1 for now to bypass a onnxruntime bug, revert it when the bug is fixed
class GRUTests(Tf2OnnxBackendTestBase):
    def test_single_dynamic_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")

        # no scope
        cell = rnn.GRUCell(
            units,
            activation=None)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_multiple_dynamic_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        _ = tf.placeholder(tf.float32, x_val.shape, name="input_2")

        gru_output_list = []
        gru_cell_state_list = []
        # no scope
        cell = rnn.GRUCell(
            units,
            activation=None)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)
        gru_output_list.append(outputs)
        gru_cell_state_list.append(cell_state)

        # given scope
        cell = rnn.GRUCell(
            units,
            activation=None)
        with variable_scope.variable_scope("root1") as scope:
            outputs, cell_state = tf.nn.dynamic_rnn(
                cell,
                x,
                dtype=tf.float32,
                sequence_length=[4],
                scope=scope)
        gru_output_list.append(outputs)
        gru_cell_state_list.append(cell_state)

        _ = tf.identity(gru_output_list, name="output")
        _ = tf.identity(gru_cell_state_list, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 2))

    def test_single_dynamic_gru_seq_length_is_const(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32,
            sequence_length=[5])

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_single_dynamic_gru_seq_length_is_not_const(self):
        for np_dtype, tf_dtype in [[np.int32, tf.int32], [np.int64, tf.int64], [np.float32, tf.float32]]:
            tf.reset_default_graph()
            units = 5
            batch_size = 1
            x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
            x_val = np.stack([x_val] * batch_size)
            x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
            initializer = init_ops.constant_initializer(0.5)

            y_val = np.array([5], dtype=np_dtype)
            seq_length = tf.placeholder(tf_dtype, y_val.shape, name="input_2")

            # no scope
            cell = rnn.GRUCell(
                units,
                kernel_initializer=initializer)
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
            self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                               graph_validator=lambda g: check_gru_count(g, 1))

    def test_single_dynamic_gru_placeholder_input(self):
        units = 5
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * 1)
        x = tf.placeholder(tf.float32, shape=(None, 4, 2), name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)  # by default zero initializer is used

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_single_dynamic_gru_ch_zero_state_initializer(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.GRUCell(
            units,
            kernel_initializer=initializer)

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
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_single_dynamic_gru_random_weights(self):
        hidden_size = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = tf.random_uniform_initializer(-1.0, 1.0)

        # no scope
        cell = rnn.GRUCell(
            hidden_size,
            kernel_initializer=initializer)

        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.0001,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_single_dynamic_gru_random_weights2(self):
        hidden_size = 128
        batch_size = 1
        x_val = np.random.randn(1, 133).astype('f')
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = tf.random_uniform_initializer(0.0, 1.0)
        # no scope
        cell = rnn.GRUCell(
            hidden_size,
            kernel_initializer=initializer)

        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.01,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_gru_output_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = tf.random_uniform_initializer(-1.0, 1.0)
        cell1 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)

        outputs, _ = tf.nn.dynamic_rnn(
            cell1,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.0001,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_gru_state_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = tf.random_uniform_initializer(-1.0, 1.0)
        cell1 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)

        _, cell_state = tf.nn.dynamic_rnn(
            cell1,
            x,
            dtype=tf.float32)

        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=0.0001, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_bigru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bigru, no scope
        cell1 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        cell2 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
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
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_bigru_output_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bigru, no scope
        cell1 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        cell2 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell1,
            cell2,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_bigru_state_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bigru, no scope
        cell1 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        cell2 = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        _, cell_state = tf.nn.bidirectional_dynamic_rnn(
            cell1,
            cell2,
            x,
            dtype=tf.float32)

        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_bidirectional_but_one_gru(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bigru, no scope
        cell = rnn.GRUCell(
            units,
            kernel_initializer=initializer)
        outputs, cell_state = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_bidirectional_but_one_gru_and_output_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")

        # bigru, no scope
        cell = rnn.GRUCell(
            units)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))

    def test_dynamic_bidirectional_but_one_gru_and_state_consumed_only(self):
        units = 5
        batch_size = 1
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")

        # bigru, no scope
        cell = rnn.GRUCell(
            units)
        _, cell_state = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_gru_count(g, 1))


if __name__ == '__main__':
    unittest_main()
