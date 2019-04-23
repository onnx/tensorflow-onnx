# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for shape inference."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from onnx import TensorProto
from tf2onnx import utils
from tf2onnx.graph import Graph
from backend_test_base import Tf2OnnxBackendTestBase
from common import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pylint: disable=missing-docstring

INPUT1 = "input1"
INPUT2 = "input2"
INPUT3 = "input3"


class ShapeInferenceTests(Tf2OnnxBackendTestBase):
    """
    Test shape inference, it's just a subset of all cases that can be inferred shape.
    For more information, please refer to onnx shape inference test:
    https://github.com/onnx/onnx/blob/master/onnx/test/shape_inference_test.py
    """

    def _run_test_case(self, graph, feed_dict):
        """Run model with onnxruntime and compare results' shape with internal shape inference."""
        outputs = graph.outputs
        results = self.run_backend(graph, outputs, feed_dict)

        for actual, inferred in zip(results, outputs):
            actual_shape = actual.shape
            inferred_shape = tuple(graph.get_shape(inferred))
            self.assertTrue(utils.are_shapes_compatible(actual_shape, inferred_shape))

            actual_dtype = actual.dtype
            inferred_dtype = utils.ONNX_TO_NUMPY_DTYPE[graph.get_dtype(inferred)]
            self.assertEqual(actual_dtype, inferred_dtype)

    def _create_empty_graph(self, inputs, shapes, dtypes):
        graph = Graph([], target=self.config.target, opset=self.config.opset)
        for inp, shape, dtype in zip(inputs, shapes, dtypes):
            graph.add_graph_input(inp, dtype, shape)
        return graph

    def _generate_random_inputs(self, inputs, shapes, dtypes):
        """Generate random input of shape and ONNX dtypes"""
        res = {}
        for inp, shape, dtype in zip(inputs, shapes, dtypes):
            new_shape = [1 if s == -1 else s for s in shape]
            res[inp] = np.random.random(new_shape).astype(utils.ONNX_TO_NUMPY_DTYPE[dtype])
        return res

    # one input
    def test_identity(self):
        inputs = [INPUT1]
        shapes = [[1, 3, 4, 1]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("Identity", [INPUT1])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_transpose(self):
        inputs = [INPUT1]
        shapes = [[1, 3, 4, 1]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("Transpose", [INPUT1], attr={"perm": [1, 0, 2, 3]})
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_shape(self):
        inputs = [INPUT1]
        shapes = [[1, 3, 4, 1]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("Shape", [INPUT1])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    @check_opset_min_version(2, "Split")
    def test_split(self):
        inputs = [INPUT1]
        shapes = [[5, 6, 7]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("Split", [INPUT1], output_count=2, attr={"axis": 1})
        graph.add_graph_output(node.output[0])
        graph.add_graph_output(node.output[1])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    # two inputs
    @check_opset_min_version(6, "Add")
    def test_add(self):
        inputs = [INPUT1, INPUT2]
        shapes = [[1, 3, 4, 1], [2, 1, 4, 10]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("Add", [INPUT1, INPUT2])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_matmul(self):
        inputs = [INPUT1, INPUT2]
        shapes = [[1, 3, 4, 1], [2, 1, 1, 10]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("MatMul", [INPUT1, INPUT2])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def _test_matmul_unknown_shape(self, shapes):
        data_shapes = [
            [1 if s == -1 else s for s in shapes[0]],
            [1 if s == -1 else s for s in shapes[1]]
        ]
        inputs = [INPUT1, INPUT2]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("MatMul", [INPUT1, INPUT2])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, data_shapes, dtypes))

    def test_matmul_unknown(self):
        self._test_matmul_unknown_shape([[-1], [-1]])
        self._test_matmul_unknown_shape([[3], [-1]])
        self._test_matmul_unknown_shape([[2], [2, -1]])
        self._test_matmul_unknown_shape([[4, 2], [2, -1]])
        self._test_matmul_unknown_shape([[1, 4, 2], [-1, 2, 5]])
        self._test_matmul_unknown_shape([[1, 3, 4, 2], [-1, 2, 5]])

    @check_opset_min_version(4, "Concat")
    def test_concat(self):
        inputs = [INPUT1, INPUT2]
        shapes = [[10, 20, 9], [12, 20, 9]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("Concat", [INPUT1, INPUT2], attr={"axis": 0})
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_conv(self):
        inputs = [INPUT1, INPUT2]
        shapes = [[3, 4, 5, 6, 7], [5, 4, 2, 4, 3]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node(
            "Conv", [INPUT1, INPUT2],
            attr={
                "pads": [0, 1, 1, 0, 0, 1],
                "dilations": [1, 2, 2],
                "strides": [1, 1, 2]
            }
        )
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    # more than two inputs
    @check_opset_min_version(8, "Sum")
    def test_sum(self):
        inputs = [INPUT1, INPUT2, INPUT3]
        shapes = [[30, 1, 5], [-1, 4, 1], [4, -1]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("Sum", [INPUT1, INPUT2, INPUT3])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    @check_opset_min_version(7, "RNN")
    def test_rnn(self):
        seq_len = 64
        batch_size = 32
        input_size = 10
        hidden_size = 4
        inputs = [INPUT1, INPUT2, INPUT3]
        shapes = [[seq_len, batch_size, input_size], [1, hidden_size, input_size], [1, hidden_size, hidden_size]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("RNN", [INPUT1, INPUT2, INPUT3], output_count=2, attr={"hidden_size": hidden_size})
        graph.add_graph_output(node.output[0])
        graph.add_graph_output(node.output[1])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    @check_opset_min_version(7, "LSTM")
    def test_lstm(self):
        seq_len = 64
        batch_size = 32
        input_size = 10
        hidden_size = 4
        inputs = [INPUT1, INPUT2, INPUT3]
        shapes = [
            [seq_len, batch_size, input_size],
            [1, 4 * hidden_size, input_size],
            [1, 4 * hidden_size, hidden_size]
        ]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("LSTM", [INPUT1, INPUT2, INPUT3], output_count=3, attr={"hidden_size": hidden_size})
        graph.add_graph_output(node.output[0])
        graph.add_graph_output(node.output[1])
        graph.add_graph_output(node.output[2])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    # with const input
    @check_opset_min_version(5, "Reshape")
    def test_reshape(self):
        inputs = [INPUT1]
        shapes = [[10, 20]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        const = graph.make_const("shape", np.array([5, 40], dtype=np.int64))
        node = graph.make_node("Reshape", [INPUT1, const.output[0]])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_gather(self):
        inputs = [INPUT1]
        shapes = [[4, 3]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        const = graph.make_const("index", np.array([1, 2], dtype=np.int64))
        node = graph.make_node("Gather", [INPUT1, const.output[0]])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_gather_axis1(self):
        inputs = [INPUT1]
        shapes = [[4, 3, 5]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        const = graph.make_const("index", np.array([[1, 2]], dtype=np.int64))
        node = graph.make_node("Gather", [INPUT1, const.output[0]], attr={"axis": 1})
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_gather_into_scalar(self):
        inputs = [INPUT1]
        shapes = [[4]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        const = graph.make_const("index", np.array(2, dtype=np.int64))
        node = graph.make_node("Gather", [INPUT1, const.output[0]])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    @check_opset_min_version(9, "ConstantOfShape")
    def test_constant_of_shape(self):
        inputs = [INPUT1]
        shapes = [[3]]
        dtypes = [TensorProto.INT64]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        node = graph.make_node("ConstantOfShape", [INPUT1])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

        graph = self._create_empty_graph([], [], [])
        const = graph.make_const("shape", np.array([3, 5, 6], dtype=np.int64))
        node = graph.make_node("ConstantOfShape", [const.output[0]])
        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs([], [], []))

    # node with subgraph
    @check_opset_min_version(8, "Scan")
    @skip_opset(9, "Scan")
    def test_scan(self):
        batch_size = 1
        seq_len = 16
        input_size = 2
        loop_state_size = 3
        inputs = [INPUT1, INPUT2]
        shapes = [[batch_size, loop_state_size, loop_state_size],
                  [batch_size, seq_len, input_size, input_size]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]

        graph = self._create_empty_graph(inputs, shapes, dtypes)

        subgraph = self._create_empty_graph(
            ["loop_state_in", "input"],
            [[-1, -1], [-1, -1]],
            [TensorProto.FLOAT, TensorProto.FLOAT],
        )
        subgraph.parent_graph = graph
        loop_state_iden = subgraph.make_node("Identity", ["loop_state_in"])
        input_iden = subgraph.make_node("Identity", ["input"])
        subgraph.add_graph_output(loop_state_iden.output[0])
        subgraph.add_graph_output(input_iden.output[0])

        seq_len_node = graph.make_const("seq_len", np.array(seq_len, dtype=np.int64))
        scan = graph.make_node(
            "Scan", [seq_len_node.output[0], INPUT1, INPUT2],
            output_count=2, attr={"num_scan_inputs": 1}
        )
        scan.set_body_graph_as_attr("body", subgraph)

        # explicitly infer shape for scan node
        graph.update_node_shape_dtype(scan)

        graph.add_graph_output(scan.output[0])
        graph.add_graph_output(scan.output[1])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    @check_opset_min_version(9, "Scan")
    def test_scan_opset9(self):
        seq_len = 16
        input_size = 2
        loop_state_size = 3
        inputs = [INPUT1, INPUT2]
        shapes = [[loop_state_size, loop_state_size],
                  [seq_len, input_size, input_size]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)

        subgraph = self._create_empty_graph(
            ["loop_state_in", "input"],
            [[-1, -1], [-1, -1]],
            [TensorProto.FLOAT, TensorProto.FLOAT],
        )
        subgraph.parent_graph = graph
        loop_state_iden = subgraph.make_node("Identity", ["loop_state_in"])
        input_iden = subgraph.make_node("Identity", ["input"])
        subgraph.add_graph_output(loop_state_iden.output[0])
        subgraph.add_graph_output(input_iden.output[0])

        scan = graph.make_node("Scan", [INPUT1, INPUT2], output_count=2, attr={"num_scan_inputs": 1})
        scan.set_body_graph_as_attr("body", subgraph)

        # explicitly infer shape for scan node
        graph.update_node_shape_dtype(scan)

        graph.add_graph_output(scan.output[0])
        graph.add_graph_output(scan.output[1])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_if(self):
        inputs = [INPUT1, INPUT2, INPUT3]
        shapes = [[2, 3, 4], [2, 3, 4], [2, 3, 4]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)

        then_subgraph = self._create_empty_graph([], [], [])
        then_subgraph.parent_graph = graph
        add = then_subgraph.make_node("Add", [INPUT1, INPUT2])
        then_subgraph.add_graph_output(add.output[0])

        else_subgraph = self._create_empty_graph([], [], [])
        else_subgraph.parent_graph = graph
        sub = else_subgraph.make_node("Sub", [INPUT1, INPUT3])
        else_subgraph.add_graph_output(sub.output[0])

        cond = graph.make_const("cond", np.array(True, dtype=np.bool))
        if_node = graph.make_node("If", [cond.output[0]])
        if_node.set_body_graph_as_attr("then_branch", then_subgraph)
        if_node.set_body_graph_as_attr("else_branch", else_subgraph)

        # explicitly infer shape for if node
        graph.update_node_shape_dtype(if_node)

        graph.add_graph_output(if_node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    def test_loop(self):
        inputs = [INPUT1, INPUT2]
        shapes = [[3, 4], [4, 5]]
        dtypes = [TensorProto.FLOAT, TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)

        subgraph = self._create_empty_graph(
            ["iter_num", "cond", "loop_state"],
            [[-1], [-1], [-1, -1]],
            [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT]
        )
        subgraph.parent_graph = graph
        cond_out = subgraph.make_node("Identity", ["cond"])
        loop_state_out = subgraph.make_node("Identity", ["loop_state"])
        out = subgraph.make_node("MatMul", ["loop_state", INPUT2])
        subgraph.add_graph_output(cond_out.output[0])
        subgraph.add_graph_output(loop_state_out.output[0])
        subgraph.add_graph_output(out.output[0])

        max_iter = graph.make_const("max_iter", np.array([10], dtype=np.int64))
        cond_const = graph.make_const("cond_const", np.array([True], dtype=np.bool))
        loop = graph.make_node("Loop", [max_iter.output[0], cond_const.output[0], INPUT1],
                               output_count=2)
        loop.set_body_graph_as_attr("body", subgraph)

        graph.update_node_shape_dtype(loop)

        # state shape may change between iterations
        # graph.add_graph_output(loop.output[0])
        graph.add_graph_output(loop.output[1])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))

    # overrider shape
    def test_override_shape(self):
        inputs = [INPUT1]
        shapes = [[1, 3, 4, 1]]
        dtypes = [TensorProto.FLOAT]
        graph = self._create_empty_graph(inputs, shapes, dtypes)
        output_name = utils.make_name("output")
        graph._output_shapes[output_name] = [-1, -1, 2, 3]  # pylint: disable=protected-access
        node = graph.make_node("Transpose", [INPUT1], attr={"perm": [1, 0, 2, 3]}, outputs=[output_name])

        graph.update_node_shape_dtype(node, override=True)

        graph.add_graph_output(node.output[0])
        self._run_test_case(graph, self._generate_random_inputs(inputs, shapes, dtypes))


if __name__ == "__main__":
    unittest_main()
