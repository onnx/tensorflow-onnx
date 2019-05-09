# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for Tensorflow shape inference."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import init_ops

from backend_test_base import Tf2OnnxBackendTestBase
from common import *  # pylint: disable=wildcard-import, unused-wildcard-import
from tf2onnx import utils
from tf2onnx.tfonnx import tf_optimize
from tf2onnx.shape_inference import infer_shape_for_graph

# pylint: disable=missing-docstring


class TFShapeInferenceTests(Tf2OnnxBackendTestBase):
    def _run_test_case(self, input_names_with_port, output_names_with_port):
        graph_def = None
        with tf.Session() as sess:
            # freeze graph
            origin_graph = sess.graph
            variables_lib.global_variables_initializer().run()
            output_name_without_port = [n.split(':')[0] for n in output_names_with_port]
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def,
                output_name_without_port
            )

        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')

        # optimize graph
        graph_def = tf_optimize(input_names_with_port, output_names_with_port,
                                sess.graph_def, True)

        with tf.Session() as sess:
            if self.config.is_debug_mode:
                if not os.path.exists(self.test_data_directory):
                    os.makedirs(self.test_data_directory)
                model_path = os.path.join(self.test_data_directory, self._testMethodName + "_after_tf_optimize.pb")
                utils.save_protobuf(model_path, graph_def)
                self.logger.debug("created file  %s", model_path)

        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            inferred_graph = infer_shape_for_graph(sess.graph)
            # compare each operation
            for op in origin_graph.get_operations():
                inferred_op = None
                try:
                    inferred_op = inferred_graph.get_operation_by_name(op.name)
                except KeyError:
                    continue
                self._compare_shape_for_op(op, inferred_op)

    def _compare_shape_for_op(self, op1, op2):
        """Align outputs of op2 to op1."""
        for out1, out2 in zip(op1.outputs, op2.outputs):
            expected_shape = utils.get_tf_tensor_shape(out1)
            if out1 is not None:
                actual_shape = utils.get_tf_tensor_shape(out2)
                self.assertTrue(utils.are_shapes_compatible(expected_shape, actual_shape))

    def test_while_loop_with_ta_read_and_write(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        inputs = tf.placeholder(tf.float32, (10,), name="input_2")

        inputs_2 = tf.identity(inputs)
        input_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(inputs_2)
        output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        c = lambda i, *_: tf.logical_and(tf.less(i, 10), i >= 0)

        def b(i, out_ta):
            new_i = tf.add(i, 1)
            x = input_ta.read(i)
            x = x + 3
            out_ta_new = out_ta.write(i, x)
            return new_i, out_ta_new

        i_final, out_final = tf.while_loop(c, b, [i, output_ta])
        _ = tf.identity(i_final, name="i")
        _ = tf.identity(out_final.stack(), name="output_ta")
        input_names_with_port = ["input_1:0", "input_2:0"]

        output_names_with_port = ["i:0", "output_ta:0"]
        self._run_test_case(input_names_with_port, output_names_with_port)

    def test_map_fn(self):
        def fn0(elem):
            res = elem + elem * elem
            return res

        def fn1(elem):
            res = elem[0] * elem[1] + elem[0]
            return res

        x_val = 100 * np.random.random_sample([2, 10]).astype(np.float32)
        y_val = 100 * np.random.random_sample([2, 10]).astype(np.float32)

        # test fn0
        x = tf.placeholder(tf.float32, shape=x_val.shape, name="input_0")
        x_ = tf.identity(x)
        res_ = tf.map_fn(fn0, x_, dtype=tf.float32)
        _ = tf.identity(res_, name="output_0")
        input_names_with_port = ["input_0:0"]
        output_names_with_port = ["output_0:0"]
        self._run_test_case(input_names_with_port, output_names_with_port)
        tf.reset_default_graph()

        # test fn1
        x = tf.placeholder(tf.float32, shape=x_val.shape, name="input_0")
        y = tf.placeholder(tf.float32, shape=y_val.shape, name="input_1")
        x_ = tf.identity(x)
        y_ = tf.identity(y)
        res_ = tf.map_fn(fn1, (x_, y_), dtype=tf.float32)
        _ = tf.identity(res_, name="output_0")
        input_names_with_port = ["input_0:0", "input_1:0"]
        output_names_with_port = ["output_0:0"]
        self._run_test_case(input_names_with_port, output_names_with_port)

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

        input_names_with_port = ["input_1:0", "input_2:0", "input_3:0"]
        output_names_with_port = ["output_0:0", "final_state:0"]
        self._run_test_case(input_names_with_port, output_names_with_port)

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

        cell = tf.nn.rnn_cell.LSTMCell(
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

        self._run_test_case([], output_names_with_port)

    def test_while_loop_in_cond(self):
        x_val = np.array([1, 2, 3], dtype=np.float32)
        y_val = np.array([4, 5, 6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")

        def cond_graph():
            b = tf.constant(np.array([0], dtype=np.int32), dtype=tf.int32)
            # while_loop
            c = lambda y: tf.reduce_any(tf.less(y, 10))
            b = lambda i: tf.add(y, 1)
            return tf.while_loop(c, b, [y])

        res = tf.cond(x[0] < y[0], lambda: x, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self._run_test_case(input_names_with_port, output_names_with_port)

    def test_cond_in_while_loop(self):
        i = tf.placeholder(tf.int32, (), name="input_1")
        inputs = tf.placeholder(tf.float32, (10,), name="input_2")

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
        _ = tf.identity(i_final, name="i")
        _ = tf.identity(out_final.stack(), name="output_ta")
        input_names_with_port = ["input_1:0", "input_2:0"]

        output_names_with_port = ["i:0", "output_ta:0"]
        self._run_test_case(input_names_with_port, output_names_with_port)


if __name__ == "__main__":
    unittest_main()
