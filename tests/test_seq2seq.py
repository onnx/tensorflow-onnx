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
from common import check_opset_min_version, unittest_main


# pylint: disable=missing-docstring

class Seq2SeqTests(Tf2OnnxBackendTestBase):
    def test_dynamic_decode_maximum_iterations(self):
        batch_size = 2
        num_units = 4
        vocab_size = 5
        embedding_size = 3
        GO_SYMBOL = 0
        END_SYMBOL = 1

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
            start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
            end_token=END_SYMBOL)

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


if __name__ == '__main__':
    unittest_main()
