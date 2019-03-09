# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

""" Unit Tests for tf.contrib.seq2seq """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring

class Seq2SeqTests(Tf2OnnxBackendTestBase):
    def test_dynamic_decode_maximum_iterations(self):
        batch_size = 2
        num_units = 4
        vocab_size = 5
        embedding_size = 3
        go_token = 0
        end_token = 1

        embedding = tf.constant(np.ones([vocab_size, embedding_size], dtype=np.float32))
        state_val = np.reshape([np.ones([num_units], dtype=np.float32) * i for i in range(batch_size)],
                               [batch_size, num_units])
        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(state_val, state_val)
        initializer = init_ops.constant_initializer(0.5)
        cell = rnn.LSTMCell(
            num_units=num_units,
            initializer=initializer,
            state_is_tuple=True)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=tf.tile([go_token], [batch_size]),
            end_token=end_token)

        output_layer = tf.layers.Dense(vocab_size, kernel_initializer=initializer)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=encoder_state,
            output_layer=output_layer)

        outputs, state, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=6)

        _ = tf.identity(outputs.rnn_output, name="rnn_output")
        _ = tf.identity(outputs.sample_id, name="sample_id")
        _ = tf.identity(state, name="state")
        _ = tf.identity(sequence_lengths, name="sequence_lengths")

        output_names_with_port = [
            "rnn_output:0",
            # "sample_id:0",  # incomplete type support for Transpose on onnxruntime 0.2.1
            "state:0",
        ]

        self.run_test_case({}, [], output_names_with_port, atol=1e-06, rtol=1e-6)

    def test_dynamic_decode_normal_stop(self):
        batch_size = 2
        num_units = 4
        vocab_size = 5
        embedding_size = 3
        go_token = 0
        end_token = 1

        embedding = tf.constant(np.ones([vocab_size, embedding_size], dtype=np.float32))
        state_val = np.reshape([np.ones([num_units], dtype=np.float32) * i for i in range(batch_size)],
                               [batch_size, num_units])
        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(state_val, state_val)

        cell_initializer = init_ops.constant_initializer(
            np.array([[-0.9592235, 0.42451382, 0.7437744, -0.54485345, -0.80763197,
                       0.19663906, -0.22738314, 0.7762785, 0.7464578, 0.27227187,
                       0.7661047, 0.3596425, -0.8528242, -0.89316916, -0.48946142,
                       0.87882376],
                      [0.86586094, -0.75018406, 0.25992537, -0.69368935, 0.2515502,
                       -0.26379275, 0.8954313, 0.5759742, -0.7753072, -0.4388857,
                       0.95751476, -0.82085776, -0.9467752, -0.37055635, -0.18570113,
                       -0.86504984],
                      [0.02305841, 0.3850248, 0.893692, -0.6866486, -0.83703446,
                       -0.9828961, 0.3989377, -0.59993076, 0.5330808, 0.6916566,
                       0.98468065, -0.6047034, 0.10823512, 0.34599304, -0.7834821,
                       -0.7852347],
                      [0.81643987, 0.31507468, -0.51369476, -0.12273741, 0.9701307,
                       -0.79669356, -0.34496522, -0.88750815, -0.17995334, 0.34707904,
                       -0.09201193, 0.5363934, -0.87229705, -0.5073328, -0.95894027,
                       0.5481839],
                      [-0.84093595, -0.2341497, -0.86047816, 0.43370056, -0.39073753,
                       0.37730122, 0.48026466, 0.3004985, -0.60727096, 0.9043884,
                       -0.37619448, 0.22490788, -0.03739262, 0.61672115, 0.478899,
                       -0.40780973],
                      [0.31202435, -0.22045255, -0.6087918, 0.95115066, 0.00199413,
                       -0.688287, -0.1103518, 0.4169519, 0.7913246, -0.9844644,
                       -0.6193857, 0.38659644, -0.4726901, -0.44781208, -0.5174744,
                       -0.605911],
                      [0.66771054, 0.34912825, 0.22297978, -0.4990945, 0.24057317,
                       -0.5540829, 0.92277217, 0.74939895, -0.35278273, -0.21587133,
                       -0.28613377, -0.8794241, -0.40119147, 0.67175174, -0.22741508,
                       0.37898326]], dtype=np.float32))
        dense_initializer = init_ops.constant_initializer(
            np.array([[0.56177187, -0.6233454, 0.73997784, 0.35032558, 0.6479795],
                      [0.6831174, -0.34233975, 0.39330363, 0.45177555, -0.49649096],
                      [-0.98890066, 0.6175642, 0.09800482, -0.6721206, 0.48805737],
                      [0.19671416, 0.2623148, 0.742548, 0.13555217, 0.56009054]], dtype=np.float32))

        cell = rnn.LSTMCell(
            num_units=num_units,
            initializer=cell_initializer,
            state_is_tuple=True)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=tf.tile([go_token], [batch_size]),
            end_token=end_token)

        output_layer = tf.layers.Dense(vocab_size, kernel_initializer=dense_initializer)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=encoder_state,
            output_layer=output_layer)

        outputs, state, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=6)

        _ = tf.identity(outputs.rnn_output, name="rnn_output")
        _ = tf.identity(outputs.sample_id, name="sample_id")
        _ = tf.identity(state, name="state")
        _ = tf.identity(sequence_lengths, name="sequence_lengths")

        output_names_with_port = [
            "rnn_output:0",
            # "sample_id:0",  # incomplete type support for Transpose on onnxruntime 0.2.1
            "state:0",
        ]

        self.run_test_case({}, [], output_names_with_port, atol=1e-06, rtol=1e-6)


if __name__ == '__main__':
    unittest_main()
