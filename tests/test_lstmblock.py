# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for lstm block cell."""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, check_tf_min_version, check_opset_min_version, check_lstm_count
from common import check_tf_max_version, check_opset_after_tf_version
from tf2onnx.tf_loader import is_tf2

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

if is_tf2():
    # There is no LSTMBlockCell in tf-2.x
    MultiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell
    dynamic_rnn = tf.compat.v1.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.compat.v1.nn.bidirectional_dynamic_rnn
else:
    LSTMBlockCell = tf.contrib.rnn.LSTMBlockCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell
    dynamic_rnn = tf.nn.dynamic_rnn
    bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn


class LSTMBlockTests(Tf2OnnxBackendTestBase):

    def run_test_case(self, *args, **kwargs):  #pylint: disable=arguments-differ
        # TF LSTM has an unknown dim
        tmp = self.config.allow_missing_shapes
        self.config.allow_missing_shapes = True
        try:
            super().run_test_case(*args, **kwargs)
        finally:
            self.config.allow_missing_shapes = tmp

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = LSTMBlockCell(units, use_peephole=False)
            outputs, cell_state = dynamic_rnn(cell, x, dtype=tf.float32)
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_seq_length_is_const(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = LSTMBlockCell(units)
            outputs, cell_state = dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=[4, 3, 4, 5, 2, 1])
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_seq_length_is_not_const(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        y_val = np.array([4, 3, 4, 5, 2, 1], dtype=np.int32)
        def func(x, seq_length):
            # no scope
            cell = LSTMBlockCell(units)
            outputs, cell_state = dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=tf.identity(seq_length))
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_time_major(self):
        units = 5
        seq_len = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * seq_len)
        def func(x):
            # no scope
            cell = LSTMBlockCell(units)
            outputs, cell_state = dynamic_rnn(cell, x, time_major=True, dtype=tf.float32)
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_placeholder_input(self):
        units = 5
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * 6)
        def func(x):
            # no scope
            cell = LSTMBlockCell(units)
            outputs, cell_state = dynamic_rnn(cell, x, dtype=tf.float32)  # by default zero initializer is used
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_ch_zero_state_initializer(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = LSTMBlockCell(units)
            # defining initial state
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, cell_state = dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_consume_one_of_ch_tuple(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # no scope
            cell = LSTMBlockCell(units)
            outputs, cell_state = dynamic_rnn(cell, x, dtype=tf.float32)
            return tf.identity(outputs, name="output"),\
                   tf.identity(cell_state.c, name="cell_state_c"),\
                   tf.identity(cell_state.h, name="cell_state_h")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state_c:0", "cell_state_h:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_multiple_dynamic_lstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            lstm_output_list = []
            lstm_cell_state_list = []
            # no scope
            cell = LSTMBlockCell(units)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)
            lstm_output_list.append(outputs)
            lstm_cell_state_list.append(cell_state)

            # given scope
            cell = LSTMBlockCell(units)
            with variable_scope.variable_scope("root1") as scope:
                outputs, cell_state = dynamic_rnn(
                    cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=[4, 4, 4, 4, 4, 4],
                    scope=scope)
            lstm_output_list.append(outputs)
            lstm_cell_state_list.append(cell_state)
            return tf.identity(lstm_output_list, name="output"), tf.identity(lstm_cell_state_list, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06)
        # graph_validator=lambda g: check_lstm_count(g, 2))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_lstm_output_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            cell1 = LSTMBlockCell(units)
            outputs, _ = dynamic_rnn(
                cell1,
                x,
                dtype=tf.float32)
            return tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_dynamic_lstm_state_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            cell1 = LSTMBlockCell(units)
            _, cell_state = dynamic_rnn(
                cell1,
                x,
                dtype=tf.float32)
            return tf.identity(cell_state, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06,
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # bilstm, no scope
            cell1 = LSTMBlockCell(units)
            cell2 = LSTMBlockCell(units)
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
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_output_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        def func(x):
            # bilstm, no scope
            cell1 = LSTMBlockCell(units)
            cell2 = LSTMBlockCell(units)
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
                           graph_validator=lambda g: check_lstm_count(g, 1))

    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    @check_opset_after_tf_version("1.15", 10, "might need ReverseV2")
    def test_dynamic_bilstm_state_consumed_only(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # bilstm, no scope
            cell1 = LSTMBlockCell(units)
            cell2 = LSTMBlockCell(units)
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
                           graph_validator=lambda g: check_lstm_count(g, 1))

    # ==============================================================================================
    # NOTE: the unittest above should be converted into a single LSTM op, while following unittests
    # should be first converted into a Scan op with LSTMBlockCell, then decoupled into several ops.
    # ==============================================================================================

    @check_opset_min_version(8, "Scan")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_with_peephole(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # no scope
            cell = LSTMBlockCell(units, use_peephole=True)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-3, atol=1e-06)

    @check_opset_min_version(8, "Scan")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_single_dynamic_lstm_with_cell_clip(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            # no scope
            cell = LSTMBlockCell(units, cell_clip=0.05)
            outputs, cell_state = dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)

            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06)

    @check_opset_min_version(8, "Scan")
    @check_tf_min_version("1.8")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_attention_wrapper_lstm_encoder(self):
        size = 5
        time_step = 3
        input_size = 4
        attn_size = size
        batch_size = 9
        encoder_time_step = time_step
        encoder_x_val = np.random.randn(encoder_time_step, input_size).astype('f')
        encoder_x_val = np.stack([encoder_x_val] * batch_size)
        decoder_time_step = 6
        decoder_x_val = np.random.randn(decoder_time_step, input_size).astype('f')
        decoder_x_val = np.stack([decoder_x_val] * batch_size)

        # shape  [batch size, time step, size]
        # attention_state: usually the output of an RNN encoder.
        # This tensor should be shaped `[batch_size, max_time, ...]`
        def func(encoder_x, decoder_x):
            encoder_cell = LSTMBlockCell(size)
            output, attr_state = dynamic_rnn(encoder_cell, encoder_x, dtype=tf.float32)
            output_0 = tf.identity(output, name="output_0")
            attention_states = output
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attn_size,
                                                                       attention_states)

            match_input_fn = lambda curr_input, state: tf.concat([curr_input, state], axis=-1)
            cell = LSTMBlockCell(size)
            match_cell_fw = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                                attention_mechanism,
                                                                attention_layer_size=attn_size,
                                                                cell_input_fn=match_input_fn,
                                                                output_attention=False)

            output, attr_state = dynamic_rnn(match_cell_fw, decoder_x, dtype=tf.float32)

            return output_0, tf.identity(output, name="output"), tf.identity(attr_state.cell_state, name="final_state")

        feed_dict = {"input_1:0": encoder_x_val, "input_2:0": decoder_x_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output_0:0", "output:0", "final_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, 0.1)

    @check_opset_min_version(8, "Scan")
    @check_tf_max_version("1.15", "no LSTMBlockCell in tf-2.x")
    def test_multi_rnn_lstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        def func(x):
            cell_0 = LSTMBlockCell(units)
            cell_1 = LSTMBlockCell(units)
            cell_2 = LSTMBlockCell(units)
            cells = MultiRNNCell([cell_0, cell_1, cell_2], state_is_tuple=True)
            outputs, cell_state = dynamic_rnn(cells, x, dtype=tf.float32)
            return tf.identity(outputs, name="output"), tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(func, feed_dict, input_names_with_port, output_names_with_port, rtol=1e-03, atol=1e-06)

if __name__ == '__main__':
    unittest_main()
