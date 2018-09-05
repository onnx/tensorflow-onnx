# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import unittest
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tf2onnx.utils
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tf2onnx.tfonnx import process_tf_graph

TMPPATH = tempfile.mkdtemp()

BACKEND = "caffe2"
# BACKEND = "onnxmsrt"
# BACKEND = "onnxmsrtnext"
# BACKEND = "onnx-tensorflow"

# names for input and outputs for tests
_TFINPUT = "input"
_INPUT = "input:0"
_TFINPUT1 = "input1"
_INPUT1 = "input1:0"
_TFOUTPUT = "output"
_OUTPUT = "output:0"
_OUTPUT1 = "output1:0"

OPSET = 7


class Tf2OnnxBackendTests(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        tf.reset_default_graph()
        # reset name generation on every test
        tf2onnx.utils.INTERNAL_NAME = 1
        np.random.seed(1)  # Make it reproducible.

        arg = namedtuple("Arg", "input inputs outputs verbose continue_on_error")
        self._args0 = arg(input="test", inputs=[], outputs=[_OUTPUT],
                          verbose=False, continue_on_error=False)
        self._args1 = arg(input="test", inputs=[_INPUT], outputs=[_OUTPUT],
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
    def run_onnxmsrt(onnx_graph, inputs, output_names, test_name):
        """Run test against msrt backend."""
        import lotus
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())

        m = lotus.ModelExecutor(model_path)
        results = m.run(output_names, inputs)
        return results[0]

    @staticmethod
    def run_onnxmsrtnext(onnx_graph, inputs, output_names, test_name):
        """Run test against msrt-next backend."""
        import lotus
        model_path = os.path.join(TMPPATH, test_name + ".pb")
        with open(model_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())

        m = lotus.InferenceSession(model_path)
        results = m.run(output_names, inputs)
        return results[0]

    def _run_backend(self, g, args, input_dict):
        model_proto = g.make_model("test", args.inputs, args.outputs)
        if BACKEND == "onnxmsrt":
            y = self.run_onnxmsrt(model_proto, input_dict, args.outputs, self._testMethodName)
        elif BACKEND == "onnxmsrtnext":
            y = self.run_onnxmsrtnext(model_proto, input_dict, args.outputs, self._testMethodName)
        elif BACKEND == "caffe2":
            y = self.run_onnxcaffe2(model_proto, input_dict)
        else:
            raise ValueError("unknown backend")
        return y

    def _run(self, output, tf_dict, onnx_dict):
        with tf.Session() as sess:
            expected = sess.run(output, feed_dict=tf_dict)
            g = process_tf_graph(sess.graph)
            actual = self._run_backend(g, self._args1, onnx_dict)
        return actual, expected

    def test_dynamic_lstm(self):
        units = 5
        batch_size = 6
        input_size = 2
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        initializer = init_ops.constant_initializer(0.5)

        lstm_list = []
        if True:
            # no scope
            cell = rnn.LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=True)
            outputs, cell_state = tf.nn.dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)
            lstm_list.append(outputs)

        if True:
            # given scope
            cell = rnn.LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=True)
            with variable_scope.variable_scope("root1") as scope:
                outputs, cell_state = tf.nn.dynamic_rnn(
                    cell,
                    x,
                    dtype=tf.float32,
                    scope=scope)
            lstm_list.append(outputs)

        outputs = tf.concat(lstm_list, axis=0)
        output = tf.identity(outputs, name=_TFOUTPUT)
        with tf.Session() as sess:
            variables_lib.global_variables_initializer().run()
            expected = sess.run([output], feed_dict={x: x_val})
            frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [_TFOUTPUT])
        tf.reset_default_graph()
        with tf.Session() as sess:
            tf.import_graph_def(frozen, name='')
            variables_lib.global_variables_initializer().run()
            output = sess.graph.get_tensor_by_name(_OUTPUT)
            expected = sess.run(output, feed_dict={_INPUT: x_val})
            g = process_tf_graph(sess.graph, continue_on_error=True)
            actual = self._run_backend(g, self._args1, {_INPUT: x_val})
        self.assertAllClose(expected, actual)

    def test_dynamic_bilstm(self):
        units = 5
        batch_size = 6
        input_size = 2
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name=_TFINPUT)
        initializer = init_ops.constant_initializer(0.5)

        lstm_list = []

        if True:
            # bilstm, no scope
            cell1 = rnn.LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=True)
            cell2 = rnn.LSTMCell(
                units,
                initializer=initializer,
                state_is_tuple=True)
            outputs, cell_state = tf.nn.bidirectional_dynamic_rnn(
                cell1,
                cell2,
                x,
                dtype=tf.float32)
            lstm_list.append(outputs)

        outputs = tf.concat(lstm_list, axis=0)
        output = tf.identity(outputs, name=_TFOUTPUT)
        with tf.Session() as sess:
            variables_lib.global_variables_initializer().run()
            expected = sess.run([output], feed_dict={x: x_val})
            frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [_TFOUTPUT])
        tf.reset_default_graph()
        with tf.Session() as sess:
            tf.import_graph_def(frozen, name='')
            variables_lib.global_variables_initializer().run()
            output = sess.graph.get_tensor_by_name(_OUTPUT)
            expected = sess.run(output, feed_dict={_INPUT: x_val})
            g = process_tf_graph(sess.graph, continue_on_error=True)
            actual = self._run_backend(g, self._args1, {_INPUT: x_val})
        self.assertAllClose(expected, actual)


if __name__ == "__main__":
    unittest.main()
