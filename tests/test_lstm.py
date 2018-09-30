# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import sys
import tempfile
import tensorflow as tf
import tf2onnx.utils
import unittest

from collections import namedtuple
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tf2onnx.tfonnx import process_tf_graph

TMPPATH = tempfile.mkdtemp()

# BACKEND = "caffe2"
# BACKEND = "onnxmsrt"
BACKEND = "onnxruntime"
# BACKEND = "onnx-tensorflow"

# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFOUTPUT = "output"
_TFOUTPUT_CELLSTATE = "cellstate_output"
_TFOUTPUT1 = "output1"
_OUTPUT = "output:0"
_OUTPUT_CELLSTATE = "cellstate_output:0"
_OUTPUT1 = "output1:0"

OPSET = 7

class Tf2OnnxLSTMTests(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        tf.reset_default_graph()
        # reset name generation on every test
        tf2onnx.utils.INTERNAL_NAME = 1
        np.random.seed(1)  # Make it reproducible.

        arg = namedtuple("Arg", "input inputs outputs verbose continue_on_error")
        self._args0 = arg(input="test", inputs=[], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args1 = arg(input="test", inputs=[_INPUT], outputs=[_OUTPUT, _OUTPUT_CELLSTATE],
                          verbose=False, continue_on_error=False)
        self._args2 = arg(input="test", inputs=[_INPUT, _INPUT1], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args3 = arg(input="test", inputs=[_INPUT, _INPUT1, "prob:0"], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args4 = arg(input="test", inputs=[_INPUT, _INPUT1], outputs=[_OUTPUT, _OUTPUT1],
                          verbose=False, continue_on_error=False)

    @staticmethod
    def assertAllClose(expected, actual, **kwargs):
        np.testing.assert_allclose(expected, actual, **kwargs)

    @staticmethod
    def run_onnxcaffe2(onnx_graph, inputs):
        """Run test against caffe2 backend."""
        import caffe2.python.onnx.backend
        prepared_backend = caffe2.python.onnx.backend.prepare(onnx_graph)
        results = prepared_backend.run(inputs)
        return results[0]

    @staticmethod
    def run_onnxmsrtnext(onnx_graph, inputs, output_names, test_name):
        """Run test against msrt-next backend."""
        import lotus
        model_path = os.path.join(TMPPATH, test_name + ".onnx")
        print("create model file before run in lotus: " + model_path)
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())

        m = lotus.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results

    @staticmethod
    def run_onnxruntime(onnx_graph, inputs, output_names, test_name):
        """Run test against msrt-next backend."""
        import onnxruntime as rt
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())
        m = rt.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results

    def _run_backend(self, g, outputs, input_dict):
        model_proto = g.make_model("test", outputs, False)
        if BACKEND == "onnxmsrtnext":
            y = self.run_onnxmsrtnext(model_proto, input_dict, outputs, self._testMethodName)
        elif BACKEND == "onnxruntime":
            y = self.run_onnxruntime(model_proto, input_dict, outputs, self._testMethodName)
        elif BACKEND == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y


    def _run(self, output, tf_dict, onnx_dict):
        with tf.Session() as sess:
            expected = sess.run(output, feed_dict=tf_dict)
            graph_def = tf2onnx.tfonnx.tf_optimize(None, args.inputs, args.outputs, sess.graph, True)

            outdir = "./tmp"
            os.makedirs(outdir, exist_ok=True)
            name = "tf_optmized.pb"
            model_path = os.path.join(outdir, name)
            with open(model_path, "wb") as f:
                f.write(graph_def.SerializeToString())

            print("created file " + model_path)
            g = process_tf_graph(graph_def)
            actual = self._run_backend(g, self._args1, onnx_dict)
        return actual, expected

    def run_test_temp(self, output_dict, feed_dict, input_names_with_port, output_names_with_port, rtol=0.000001):
        with tf.Session() as sess:
            variables_lib.global_variables_initializer().run()
            expected = sess.run(output_dict, feed_dict=feed_dict)

            output_name_without_port = [n.split(':')[0] for n in output_names_with_port]
            frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_name_without_port)

        tf.reset_default_graph()
        graph_def = None
        with tf.Session() as sess:
            tf.import_graph_def(frozen, name='')
            variables_lib.global_variables_initializer().run()

            output_dict = []
            for out_name in output_names_with_port:
                output_dict.append(sess.graph.get_tensor_by_name(out_name))

            expected = sess.run(output_dict, feed_dict=feed_dict)

            model_path = os.path.join(TMPPATH, "before_tf_optimize.pb")
            with open(model_path, "wb") as f:
                f.write(sess.graph_def.SerializeToString())

            print("created file " + model_path)
            graph_def = tf2onnx.tfonnx.tf_optimize(None, input_names_with_port, output_names_with_port, sess.graph_def, True)
            model_path = os.path.join(TMPPATH, "after_tf_optimize.pb")
            with open(model_path, "wb") as f:
                f.write(graph_def.SerializeToString())

            print("created file " + model_path)
        tf.reset_default_graph()
        g = tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=g) as sess:
            g = process_tf_graph(sess.graph, continue_on_error=True, enable_lstm=True) # shape_override={"output:0": [1, 6,4,5]}
            actual = self._run_backend(g, output_names_with_port, feed_dict)

        for i in range(len(expected)):
            self.assertAllClose(expected[i], actual[i], rtol=rtol, atol=0.)

    #@unittest.skip("reason for skipping")
    def test_test_single_dynamic_lstm_stateistuple(self):
        self.internel_test_single_dynamic_lstm(True)

    #@unittest.skip("reason for skipping")
    def test_test_single_dynamic_lstm_stateisnottuple(self):
        self.internel_test_single_dynamic_lstm(False)

    def internel_test_single_dynamic_lstm(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
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

        output = tf.identity(outputs, name=_TFOUTPUT)
        cellstate = tf.identity(cell_state, name=_TFOUTPUT_CELLSTATE)

        feed_dict = {_INPUT: x_val}
        output_dict = [output, cellstate]
        input_names_with_port = [_INPUT]
        output_names_with_port = [_OUTPUT, _OUTPUT_CELLSTATE]
        self.run_test_temp(output_dict, feed_dict, input_names_with_port, output_names_with_port)

    #@unittest.skip("reason for skipping")
    def test_single_dynamic_lstm_randomweights(self, state_is_tuple = True):
        hidden_size = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
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

        output = tf.identity(outputs, name=_TFOUTPUT)
        cellstate = tf.identity(cell_state, name=_TFOUTPUT_CELLSTATE)

        feed_dict = {_INPUT: x_val}
        output_dict = [output, cellstate]
        input_names_with_port = [_INPUT]
        output_names_with_port = [_OUTPUT, _OUTPUT_CELLSTATE]
        self.run_test_temp(output_dict, feed_dict, input_names_with_port, output_names_with_port, 0.0001)

    #@unittest.skip("reason for skipping")
    def test_single_dynamic_lstm_randomweights2(self, state_is_tuple = True):
        hidden_size = 128
        batch_size = 1
        x_val = np.random.randn(1, 133).astype('f')
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
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

        output = tf.identity(outputs, name=_TFOUTPUT)
        cellstate = tf.identity(cell_state, name=_TFOUTPUT_CELLSTATE)

        feed_dict = {_INPUT: x_val}
        output_dict = [output, cellstate]
        input_names_with_port = [_INPUT]
        output_names_with_port = [_OUTPUT, _OUTPUT_CELLSTATE]
        self.run_test_temp(output_dict, feed_dict, input_names_with_port, output_names_with_port, 0.01)

    #@unittest.skip("reason for skipping")
    def test_multiple_dynamic_lstm_stateistuple(self):
        self.internel_test_multiple_dynamic_lstm_with_parameters(True)

    #@unittest.skip("reason for skipping")
    def test_multiple_dynamic_lstm_stateisnottuple(self):
        self.internel_test_multiple_dynamic_lstm_with_parameters(False)

    def internel_test_multiple_dynamic_lstm_with_parameters(self, state_is_tuple):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        y = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT1)
        initializer = init_ops.constant_initializer(0.5)

        lstm_output_list = []
        lstm_cell_state_list = []
        if True:
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

        if True:
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

        output = tf.identity(lstm_output_list, name=_TFOUTPUT)
        cellstate = tf.identity(lstm_cell_state_list, name=_TFOUTPUT_CELLSTATE)

        feed_dict = {_INPUT: x_val}
        output_dict = [output, cellstate]
        input_names_with_port = [_INPUT]
        output_names_with_port = [_OUTPUT, _OUTPUT_CELLSTATE]
        self.run_test_temp(output_dict, feed_dict, input_names_with_port, output_names_with_port)

    #@unittest.skip("reason for skipping")
    def test_dynamic_basiclstm(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        cell1 = rnn.BasicLSTMCell(
            units,
            state_is_tuple=True)

        outputs, cell_state = tf.nn.dynamic_rnn(
            cell1,
            x,
            dtype=tf.float32)

        output = tf.identity(outputs, name=_TFOUTPUT)
        cellstate = tf.identity(cell_state, name=_TFOUTPUT_CELLSTATE)

        feed_dict = {_INPUT: x_val}
        output_dict = [output]
        output_dict = [output, cellstate]
        input_names_with_port = [_INPUT]
        output_names_with_port = [_OUTPUT]
        output_names_with_port = [_OUTPUT, _OUTPUT_CELLSTATE]
        self.run_test_temp(output_dict, feed_dict, input_names_with_port, output_names_with_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default=BACKEND,
                        choices=["caffe2", "onnxmsrtnext", "onnxruntime"],
                        help="backend to test against")

    parser.add_argument('--opset', type=int, default=OPSET,
                        help="opset to test against")
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    BACKEND = args.backend
    OPSET = args.opset
    # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
    sys.argv[1:] = args.unittest_args
    unittest.main()
