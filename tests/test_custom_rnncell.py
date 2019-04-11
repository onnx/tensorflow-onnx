# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for custom rnns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from backend_test_base import Tf2OnnxBackendTestBase
from common import check_tf_min_version, check_opset_min_version, unittest_main, skip_opset


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test
# pylint: disable=abstract-method,arguments-differ


class CustomRnnCellTests(Tf2OnnxBackendTestBase):
    @check_opset_min_version(8, "Scan")
    def test_single_dynamic_custom_rnn(self):
        size = 5  # size of each model layer.
        batch_size = 1
        cell = GatedGRUCell(size)

        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        xs, s = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,
                                  inputs=x, time_major=False)
        _ = tf.identity(xs, name="output")
        _ = tf.identity(s, name="final_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "final_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.1)

    @check_opset_min_version(8, "Scan")
    def test_single_dynamic_custom_rnn_time_major(self):
        size = 5  # size of each model layer.
        batch_size = 1
        cell = GatedGRUCell(size)

        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        xs, s = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,
                                  inputs=x, time_major=True)
        _ = tf.identity(xs, name="output")
        _ = tf.identity(s, name="final_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "final_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.1)

    @check_opset_min_version(8, "Scan")
    def test_single_dynamic_custom_rnn_with_seq_length(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")

        # no scope
        cell = GatedGRUCell(units)
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
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_min_version(8, "Scan")
    def test_single_dynamic_custom_rnn_with_non_const_seq_length(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")

        y_val = np.array([4, 3, 4, 5, 2, 1], dtype=np.int32)
        seq_length = tf.placeholder(tf.int32, y_val.shape, name="input_2")

        # no scope
        cell = GatedGRUCell(units)
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
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_min_version(8, "Scan")
    @check_tf_min_version("1.8")
    def test_attention_wrapper_const_encoder(self):
        size = 5
        time_step = 3
        input_size = 4
        attn_size = size

        batch_size = 9
        # shape  [batch size, time step, size]
        # attention_state: usually the output of an RNN encoder.
        # This tensor should be shaped `[batch_size, max_time, ...]`.
        attention_states = np.random.randn(batch_size, time_step, attn_size).astype('f')
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            attn_size,
            attention_states)

        match_input_fn = lambda curr_input, state: tf.concat([curr_input, state], axis=-1)
        cell = tf.nn.rnn_cell.LSTMCell(size)
        match_cell_fw = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                            attention_mechanism,
                                                            attention_layer_size=attn_size,
                                                            cell_input_fn=match_input_fn,
                                                            output_attention=False)

        decoder_time_step = 6
        x_val = np.random.randn(decoder_time_step, input_size).astype('f')
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        output, attr_state = tf.nn.dynamic_rnn(match_cell_fw, x, dtype=tf.float32)

        _ = tf.identity(output, name="output")
        _ = tf.identity(attr_state.cell_state, name="final_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        output_names_with_port = ["output:0", "final_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.1)

    @check_opset_min_version(8, "Scan")
    @check_tf_min_version("1.8")
    def test_attention_wrapper_lstm_encoder(self):
        size = 5
        time_step = 3
        input_size = 4
        attn_size = size

        batch_size = 9

        # shape  [batch size, time step, size]
        # attention_state: usually the output of an RNN encoder.
        # This tensor should be shaped `[batch_size, max_time, ...]`
        encoder_time_step = time_step
        encoder_x_val = np.random.randn(encoder_time_step, input_size).astype('f')
        encoder_x_val = np.stack([encoder_x_val] * batch_size)
        encoder_x = tf.placeholder(tf.float32, encoder_x_val.shape, name="input_1")
        encoder_cell = tf.nn.rnn_cell.LSTMCell(size)
        output, attr_state = tf.nn.dynamic_rnn(encoder_cell, encoder_x, dtype=tf.float32)
        _ = tf.identity(output, name="output_0")
        attention_states = output
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attn_size,
                                                                   attention_states)

        match_input_fn = lambda curr_input, state: tf.concat([curr_input, state], axis=-1)
        cell = tf.nn.rnn_cell.LSTMCell(size)
        match_cell_fw = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                            attention_mechanism,
                                                            attention_layer_size=attn_size,
                                                            cell_input_fn=match_input_fn,
                                                            output_attention=False)

        decoder_time_step = 6
        decoder_x_val = np.random.randn(decoder_time_step, input_size).astype('f')
        decoder_x_val = np.stack([decoder_x_val] * batch_size)

        decoder_x = tf.placeholder(tf.float32, decoder_x_val.shape, name="input_2")
        output, attr_state = tf.nn.dynamic_rnn(match_cell_fw, decoder_x, dtype=tf.float32)

        _ = tf.identity(output, name="output")
        _ = tf.identity(attr_state.cell_state, name="final_state")

        feed_dict = {"input_1:0": encoder_x_val, "input_2:0": decoder_x_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output_0:0", "output:0", "final_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.1)

    @check_opset_min_version(8, "Scan")
    @check_tf_min_version("1.8")
    def test_attention_wrapper_gru_encoder(self):
        size = 5
        time_step = 3
        input_size = 4
        attn_size = size

        batch_size = 9

        # shape  [batch size, time step, size]
        # attention_state: usually the output of an RNN encoder.
        # This tensor should be shaped `[batch_size, max_time, ...]`
        encoder_time_step = time_step
        encoder_x_val = np.random.randn(encoder_time_step, input_size).astype('f')
        encoder_x_val = np.stack([encoder_x_val] * batch_size)
        encoder_x = tf.placeholder(tf.float32, encoder_x_val.shape, name="input_1")
        encoder_cell = tf.nn.rnn_cell.GRUCell(size)
        output, attr_state = tf.nn.dynamic_rnn(encoder_cell, encoder_x, dtype=tf.float32)
        _ = tf.identity(output, name="output_0")
        attention_states = output
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attn_size,
                                                                   attention_states)

        match_input_fn = lambda curr_input, state: tf.concat([curr_input, state], axis=-1)
        cell = tf.nn.rnn_cell.GRUCell(size)
        match_cell_fw = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                            attention_mechanism,
                                                            attention_layer_size=attn_size,
                                                            cell_input_fn=match_input_fn,
                                                            output_attention=False)

        decoder_time_step = 6
        decoder_x_val = np.random.randn(decoder_time_step, input_size).astype('f')
        decoder_x_val = np.stack([decoder_x_val] * batch_size)

        decoder_x = tf.placeholder(tf.float32, decoder_x_val.shape, name="input_2")
        output, attr_state = tf.nn.dynamic_rnn(match_cell_fw, decoder_x, dtype=tf.float32)

        _ = tf.identity(output, name="output")
        _ = tf.identity(attr_state.cell_state, name="final_state")

        feed_dict = {"input_1:0": encoder_x_val, "input_2:0": decoder_x_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output_0:0", "output:0", "final_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.1)

    @check_opset_min_version(8, "Scan")
    @check_tf_min_version("1.8")
    def test_attention_wrapper_lstm_encoder_input_has_none_dim(self):
        size = 5
        time_step = 3
        input_size = 4
        attn_size = size

        batch_size = 9

        # shape  [batch size, time step, size]
        # attention_state: usually the output of an RNN encoder.
        # This tensor should be shaped `[batch_size, max_time, ...]`
        encoder_time_step = time_step
        encoder_x_val = np.random.randn(encoder_time_step, input_size).astype('f')
        encoder_x = tf.placeholder(tf.float32, (batch_size,) + encoder_x_val.shape, name="input_1")
        encoder_x_val = np.stack([encoder_x_val] * batch_size)
        encoder_cell = tf.nn.rnn_cell.LSTMCell(size)
        output, attr_state = tf.nn.dynamic_rnn(encoder_cell, encoder_x, dtype=tf.float32)
        _ = tf.identity(output, name="output_0")
        attention_states = output
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attn_size,
                                                                   attention_states)

        match_input_fn = lambda curr_input, state: tf.concat([curr_input, state], axis=-1)
        cell = tf.nn.rnn_cell.LSTMCell(size)
        match_cell_fw = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                            attention_mechanism,
                                                            attention_layer_size=attn_size,
                                                            cell_input_fn=match_input_fn,
                                                            output_attention=False)

        decoder_time_step = 6
        decoder_x_val = np.random.randn(decoder_time_step, input_size).astype('f')
        decoder_x = tf.placeholder(tf.float32, (None,) + decoder_x_val.shape, name="input_2")
        decoder_x_val = np.stack([decoder_x_val] * batch_size)

        output, attr_state = tf.nn.dynamic_rnn(match_cell_fw, decoder_x, dtype=tf.float32)

        _ = tf.identity(output, name="output")
        _ = tf.identity(attr_state.cell_state, name="final_state")

        feed_dict = {"input_1:0": encoder_x_val, "input_2:0": decoder_x_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output_0:0", "output:0", "final_state:0"]

        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.1)

    @check_opset_min_version(8, "Scan")
    def test_multi_rnn_lstm(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        cell_0 = rnn.LSTMCell(units,
                              initializer=initializer,
                              state_is_tuple=state_is_tuple)

        cell_1 = rnn.LSTMCell(units,
                              initializer=initializer,
                              state_is_tuple=state_is_tuple)

        cell_2 = rnn.LSTMCell(units,
                              initializer=initializer,
                              state_is_tuple=state_is_tuple)

        cells = rnn.MultiRNNCell([cell_0, cell_1, cell_2], state_is_tuple=state_is_tuple)
        outputs, cell_state = tf.nn.dynamic_rnn(cells,
                                                x,
                                                dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)

    @check_opset_min_version(8, "Scan")
    @check_tf_min_version("1.8")
    @skip_opset(9, "ReverseSequence")
    def test_bidrectional_attention_wrapper_lstm_encoder(self):
        size = 30
        time_step = 3
        input_size = 4
        attn_size = size
        batch_size = 9

        # shape  [batch size, time step, size]
        # attention_state: usually the output of an RNN encoder.
        # This tensor should be shaped `[batch_size, max_time, ...]`
        encoder_time_step = time_step
        encoder_x_val = np.random.randn(encoder_time_step, input_size).astype('f')
        encoder_x_val = np.stack([encoder_x_val] * batch_size)
        encoder_x = tf.placeholder(tf.float32, encoder_x_val.shape, name="input_1")
        encoder_cell = tf.nn.rnn_cell.LSTMCell(size)
        attention_states, _ = tf.nn.dynamic_rnn(encoder_cell, encoder_x, dtype=tf.float32)
        # [9, 3, 30], [9, 30]
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attn_size,
                                                                   attention_states)

        match_input_fn = lambda curr_input, state: tf.concat([curr_input, state], axis=-1)
        cell = tf.nn.rnn_cell.LSTMCell(size)
        match_cell_fw = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                            attention_mechanism,
                                                            attention_layer_size=attn_size,
                                                            cell_input_fn=match_input_fn,
                                                            output_attention=False)
        match_cell_bk = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                            attention_mechanism,
                                                            attention_layer_size=attn_size,
                                                            cell_input_fn=match_input_fn,
                                                            output_attention=False)

        decoder_time_step = 6
        decoder_x_val = np.random.randn(decoder_time_step, batch_size, input_size).astype('f')

        decoder_x = tf.placeholder(tf.float32, decoder_x_val.shape, name="input_2")
        seq_length = tf.placeholder(tf.int32, (batch_size), name="input_3")
        (match_output_fw, match_output_bk), (match_state_fw, match_state_bk) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=match_cell_fw,
                                            cell_bw=match_cell_bk,
                                            inputs=decoder_x,
                                            sequence_length=tf.identity(seq_length),
                                            dtype=tf.float32,
                                            time_major=True)

        matched_output = tf.concat([match_output_fw, match_output_bk], axis=-1)
        matched_state = tf.concat([match_state_fw.cell_state, match_state_bk.cell_state], -1)

        _ = tf.identity(matched_output, name="output_0")
        _ = tf.identity(matched_state, name="final_state")

        feed_dict = {"input_1:0": encoder_x_val, "input_2:0": decoder_x_val,
                     "input_3:0": np.array([6, 5, 4, 3, 2, 1, 2, 3, 6], dtype=np.int32)}
        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        output_names_with_port = ["output_0:0", "final_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, 0.1)


class GatedGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, hidden_dim, reuse=None):
        super().__init__(self, _reuse=reuse)
        self._num_units = hidden_dim
        self._activation = tf.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        # inputs shape: [batch size, time step, input size] = [1, 3, 2]
        # num_units: 5
        # W shape: [2, 3 * 5] = [2, 15]
        # U shape: [5, 3 * 5] = [5, 15]
        # b shape: [1, 3 * 5] = [1, 15]
        # state shape: [batch size, state size] = [1, 5]

        input_dim = inputs.get_shape()[-1]
        assert input_dim is not None, "input dimension must be defined"
        # W = tf.get_variable(name="W", shape=[input_dim, 3 * self._num_units], dtype=tf.float32)
        W = np.arange(30.0, dtype=np.float32).reshape(2, 15)
        # U = tf.get_variable(name='U', shape=[self._num_units, 3 * self._num_units], dtype=tf.float32)
        U = np.arange(75.0, dtype=np.float32).reshape(5, 15)
        # b = tf.get_variable(name='b', shape=[1, 3 * self._num_units], dtype=tf.float32)
        b = np.arange(15.0, dtype=np.float32).reshape(1, 15)

        xw = tf.split(tf.matmul(inputs, W) + b, 3, 1)
        hu = tf.split(tf.matmul(state, U), 3, 1)
        r = tf.sigmoid(xw[0] + hu[0])
        z = tf.sigmoid(xw[1] + hu[1])
        h1 = self._activation(xw[2] + r * hu[2])
        next_h = h1 * (1 - z) + state * z
        return next_h, next_h


if __name__ == '__main__':
    unittest_main()
