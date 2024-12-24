# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for TFLite utils."""

import os
import tensorflow as tf

from common import *  # pylint: disable=wildcard-import,unused-wildcard-import
from backend_test_base import Tf2OnnxBackendTestBase
from tf2onnx.tf_loader import from_function, tf_session
from tf2onnx.tflite_utils import read_tflite_model, parse_tflite_graph

# pylint: disable=missing-docstring


class TFListUtilsTests(Tf2OnnxBackendTestBase):

    @check_tf_min_version("2.0")
    def test_parse_tflite_graph(self):

        def func(a, b, c):
            alpha = tf.constant(1.1, dtype=tf.float32)
            beta = tf.constant(2.3, dtype=tf.float32)
            mul1 = tf.multiply(alpha, tf.matmul(a, b))
            mul2 = tf.multiply(beta, c)
            x_ = mul1 + mul2
            return tf.identity(x_, name="output")

        inp_shapes = [[2, 3], [3, 1], [2, 1]]
        inp_dtypes = [tf.float32, tf.float32, tf.float32]
        names = ['a', 'b', 'c']
        names_with_port = ['a:0', 'b:0', 'c:0']
        output_names = ['output']
        output_names_with_port = ['output:0']

        input_tensors = [tf.TensorSpec(shape=s, dtype=d, name=n) for s, d, n in zip(inp_shapes, inp_dtypes, names)]

        concrete_func = tf.function(func, input_signature=tuple(input_tensors))
        concrete_func = concrete_func.get_concrete_function()
        graph_def = from_function(concrete_func,
                                  input_names=names_with_port,
                                  output_names=output_names_with_port)
        with tf_session() as sess:
            tf.import_graph_def(graph_def, name='')
            sess_inputs = [sess.graph.get_tensor_by_name(k) for k in names_with_port]
            sess_outputs = [sess.graph.get_tensor_by_name(n) for n in output_names_with_port]
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, sess_inputs, sess_outputs)

        tflite_model = converter.convert()
        tflite_path = os.path.join(self.test_data_directory, self._testMethodName + ".tflite")
        dir_name = os.path.dirname(tflite_path)
        tflite_model = converter.convert()
        os.makedirs(dir_name, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        tflite_graphs, opcodes_map, model, tensor_shapes = read_tflite_model(tflite_path)
        self.assertEqual(1, len(tflite_graphs))
        onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, inputs, outputs, _ = \
            parse_tflite_graph(tflite_graphs[0], opcodes_map, model, tensor_shapes_override=tensor_shapes)
        self.assertEqual(2, op_cnt['TFL_MUL'])
        self.assertEqual(1, op_cnt['TFL_ADD'])

        self.assertEqual(1, attr_cnt['PotScaleInt16'])
        self.assertEqual(names, inputs)
        self.assertEqual(output_names, outputs)

        for name, shape, dtype in zip(names, inp_shapes, inp_dtypes):
            self.assertEqual(shape, output_shapes[name])
            self.assertEqual(dtype, dtypes[name])

        self.assertTrue(len(onnx_nodes) >= 4)
