# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for optimizers such as TransposeOptimizer."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import helper, TensorProto
from tf2onnx.graph import GraphUtil
from backend_test_base import Tf2OnnxBackendTestBase


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


class OptimizerTests(Tf2OnnxBackendTestBase):
    """Test cases."""
    def run_and_compare(self, output_names_with_port, onnx_feed_dict, origin_proto, debug=False, rtol=1e-07):
        origin_model_path = self.save_onnx_model(origin_proto, onnx_feed_dict, postfix="_origin")

        new_proto = GraphUtil.opt_transposes_with_model_proto(origin_proto, output_names=output_names_with_port)
        new_model_path = self.save_onnx_model(new_proto, onnx_feed_dict, postfix="_opt")

        previous = GraphUtil.get_node_count_from_onnx_graph(origin_proto.graph)
        current = GraphUtil.get_node_count_from_onnx_graph(new_proto.graph)

        self.assertTrue(current["Transpose"] < previous["Transpose"], msg="transpose ops count not changed")

        if type(self).BACKEND == "onnxruntime":
            expected = self.run_onnxruntime(origin_model_path, onnx_feed_dict, output_names_with_port)
            actual = self.run_onnxruntime(new_model_path, onnx_feed_dict, output_names_with_port)
        else:
            raise ValueError("only onnxruntime is supported to test transpose optimizer")

        for expected_val, actual_val in zip(expected, actual):
            self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=0.)
            self.assertEqual(expected_val.dtype, actual_val.dtype)
            self.assertEqual(expected_val.shape, actual_val.shape)

    def test_relu(self):
        # Preprocessing: create a model with two nodes, Y's shape is unknown
        node1 = helper.make_node('Transpose', ['X'], ['Y'], perm=[0, 2, 3, 1])
        node2 = helper.make_node('Relu', ['Y'], ['Z'])
        node3 = helper.make_node('Transpose', ['Z'], ['Z1'], perm=[0, 3, 1, 2])

        graph = helper.make_graph(
            [node1, node2, node3],
            'relu-test',
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info('Z1', TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name='onnx-tests')
        self.run_and_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                             model_proto)

    def test_leaky_relu(self):
        # Preprocessing: create a model with two nodes, Y's shape is unknown
        node1 = helper.make_node('Transpose', ['X'], ['Y'], perm=[0, 2, 3, 1])
        node2 = helper.make_node('LeakyRelu', ['Y'], ['Z'], alpha=0.02)
        node3 = helper.make_node('Transpose', ['Z'], ['Z1'], perm=[0, 3, 1, 2])

        graph = helper.make_graph(
            [node1, node2, node3],
            'LeakyRelu-test',
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info('Z1', TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name='onnx-tests')
        self.run_and_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                             model_proto)

if __name__ == '__main__':
    Tf2OnnxBackendTestBase.trigger(OptimizerTests)
