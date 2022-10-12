# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for optimizers such as TransposeOptimizer."""

import unittest
import numpy as np
from onnx import helper, numpy_helper, TensorProto, OperatorSetIdProto
from parameterized import parameterized

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, group_nodes_by_type, check_opset_min_version, check_opset_max_version, get_test_config
from tf2onnx import utils, constants
from tf2onnx.graph import GraphUtil


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class OptimizerTests(Tf2OnnxBackendTestBase):
    """Run original model proto and modified model proto with onnxruntime, compare the results."""

    def run_and_compare(self, output_names_with_port, onnx_feed_dict, origin_proto, op_type,
                        remaining_op_num, debug=False, rtol=1e-07):
        utils.make_sure(op_type is not None, "op_type should be specified")
        utils.make_sure(remaining_op_num is not None, "remaining_op_num should be specified")
        utils.make_sure(self.config.is_onnxruntime_backend, "only onnxruntime is supported to test transpose optimizer")

        origin_model_path = self.save_onnx_model(origin_proto, onnx_feed_dict, postfix="_origin")
        expected = self.run_onnxruntime(origin_model_path, onnx_feed_dict, output_names_with_port)

        new_proto, new_graph = GraphUtil.optimize_model_proto(origin_proto, catch_errors=False, return_graph=True)

        self.assertTrue(new_proto, msg="model proto after optimizer should not be None")

        new_model_path = self.save_onnx_model(new_proto, onnx_feed_dict, postfix="_opt")
        current = GraphUtil.get_node_count_from_onnx_graph(new_proto.graph)

        actual = self.run_onnxruntime(new_model_path, onnx_feed_dict, output_names_with_port)

        for expected_val, actual_val in zip(expected, actual):
            self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=1e-5)
            self.assertEqual(expected_val.dtype, actual_val.dtype)
            self.assertEqual(expected_val.shape, actual_val.shape)

        self.assertTrue(current[op_type] == remaining_op_num,
                        msg="Expect " + str(remaining_op_num) + " " + op_type + " ops left, but actually " + str(
                            current[op_type]) + " left")
        self.assert_shapes_correct(new_graph, allow_missing=False, run_checker=True)

        return new_proto

    @staticmethod
    def _make_onnx_const(np_val, output_name):
        node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[output_name],
            value=helper.make_tensor(
                name=output_name,
                data_type=utils.map_numpy_to_onnx_dtype(np_val.dtype),
                dims=np_val.shape,
                vals=np_val.flatten().astype(np_val.dtype).tolist(),
            ),
        )
        return node

    def make_model(self, graph, producer_name="onnx-tests"):
        imp = OperatorSetIdProto()
        imp.version = self.config.opset
        model_proto = helper.make_model(graph, producer_name=producer_name, opset_imports=[imp])
        try:
            model_proto.ir_version = constants.OPSET_TO_IR_VERSION.get(self.config.opset, model_proto.ir_version)
        except:  # pylint: disable=bare-except
            pass
        return model_proto

    # Tranpose Optimizer Tests Start

    def run_transpose_compare(self, output_names_with_port, onnx_feed_dict, origin_proto,
                              remaining_transpose_num=None, debug=False, rtol=1e-07):
        return self.run_and_compare(output_names_with_port, onnx_feed_dict, origin_proto, op_type="Transpose",
                                    remaining_op_num=remaining_transpose_num, debug=debug, rtol=rtol)

    def check_transpose_perm(self, model_proto, expected_perm):
        for node in model_proto.graph.node:
            if node.op_type == "Transpose":
                perm = list(node.attribute[0].ints)
                self.assertEqual(perm, expected_perm)

    @parameterized.expand([
        ((2, 3, 4, 5), [0, 3, 1, 2], [0, 2, 3, 1]),
        ((2, 3, 4, 5, 6), [0, 4, 1, 2, 3], [0, 2, 3, 4, 1]),
    ])
    def test_transpose_with_concat(self, input_shape, perm, inner_perm):
        input_shape_with_trans = [input_shape[i] for i in perm]
        for axis in range(len(input_shape)):
            output_before_trans = list(input_shape)
            output_before_trans[axis] *= 2
            output_shape = [output_before_trans[i] for i in perm]
            node1 = helper.make_node("Transpose", ["input_data1"], ["Y"], perm=inner_perm, name="trans")
            node2 = helper.make_node("Concat", ["Y", "input_data2"], ["Z"], axis=axis, name="concat")
            node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm, name="trans2")

            graph = helper.make_graph(
                [node1, node2, node3],
                "test_transpose_with_concat",
                [helper.make_tensor_value_info("input_data1", TensorProto.FLOAT, input_shape_with_trans),
                 helper.make_tensor_value_info("input_data2", TensorProto.FLOAT, input_shape),
                 ],
                [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
            )

            model_proto = self.make_model(graph, producer_name="onnx-tests")
            feed_dict = {"input_data1": np.random.randn(*input_shape_with_trans).astype(np.float32),
                         "input_data2": np.random.randn(*input_shape).astype(np.float32),
                         }
            self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=1)

    @parameterized.expand([
        ((2, 3, 4, 5), [0, 3, 1, 2], [0, 2, 3, 1]),
        ((2, 3, 4, 5, 6), [0, 4, 1, 2, 3], [0, 2, 3, 4, 1]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_with_split(self, input_shape, perm, inner_perm):
        input_shape_with_trans = [input_shape[i] for i in perm]
        output_before_trans = list(input_shape)
        output_shape = [output_before_trans[i] for i in perm]
        for axis in range(len(input_shape)):
            node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=inner_perm, name="trans1")
            node2 = helper.make_node("Split", ["Y"], ["Z"], axis=axis, name="split")
            node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm, name="trans2")

            graph = helper.make_graph(
                [node1, node2, node3],
                "test_transpose_with_split",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape_with_trans)],
                [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
            )

            model_proto = self.make_model(graph, producer_name="onnx-tests")
            feed_dict = {"X": np.random.randn(*input_shape_with_trans).astype(np.float32)}
            self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((1, -1), (1, 1710), (1710,), [1, 0]),
        ((3, 1, 1, 5, -1), (3, 1, 1, 5, 6), (3, 5, 6), [0, 2, 3, 4, 1]),
    ])
    @check_opset_max_version(12, "split attribute changed to input since opset 13")
    def test_transpose_with_split_dynamic_shape(self, input_shape, specific_input, output_shape, perm):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node2 = helper.make_node("Split", ["Y"], ["Z"], axis=1, split=[1], name="split")
        node3 = helper.make_node("Squeeze", ["Z"], ["B"], name="squeeze")

        graph = helper.make_graph(
            [node1, node2, node3],
            "test_transpose_with_split_dynamic_shape",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["B"], {"X": np.random.randn(*specific_input).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 1, 1), (1, 1, 3), (1), [0, 2, 3, 1]),
        ((256, 1, 1), (1, 1, 256), (1), [0, 2, 3, 1])
    ])
    @check_opset_min_version(13, "split attribute changed to input since opset 13")
    def test_transpose_with_split_opset13(self, input_shape, output_shape, split_val, perm):
        unsqueeze_axes = self._make_onnx_const(np.array([0], dtype=np.int64), "axes1")
        unsqueeze = helper.make_node("Unsqueeze", ["X", "axes1"], ["Y"], name="unsqueeze")
        trans = helper.make_node("Transpose", ["Y"], ["Z"], perm=perm, name="trans")
        split_attr = self._make_onnx_const(np.array([split_val], dtype=np.int64), "split_attr")
        split = helper.make_node("Split", ["Z", "split_attr"], ["A"], axis=0, name="split")
        squeeze_axes = self._make_onnx_const(np.array([1], dtype=np.int64), "axes2")
        squeeze = helper.make_node("Squeeze", ["A", "axes2"], ["B"], name="squeeze")

        graph = helper.make_graph(
            [unsqueeze_axes, unsqueeze, trans, split_attr, split, squeeze_axes, squeeze],
            "test_transpose_with_split_opset13",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["B"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_abs(self, shape, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans1")
        node1 = helper.make_node("Abs", ["Y"], ["Z"], name="abs")
        node2 = helper.make_node("Transpose", ["Z"], ["OUT"], perm=perm_output, name="trans2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-abs-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_with_add1(self, input_shape, perm_input, perm_output):
        # when transpose follows with a broadcasting op
        # reshape is needed when switching transpose with this op and op need broadcast its inputs
        node1 = helper.make_node("Transpose", ["input_data1"], ["Y"], perm=perm_input, name="trans")
        node2 = helper.make_node("Add", ["Y", "input_data2"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "transpose_with_shape",
            [helper.make_tensor_value_info("input_data1", TensorProto.FLOAT, input_shape),
             helper.make_tensor_value_info("input_data2", TensorProto.FLOAT, (input_shape[1],)),
             ],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        feed_dict = {"input_data1": np.random.randn(*input_shape).astype(np.float32),
                     "input_data2": np.random.randn(input_shape[1]).astype(np.float32),
                     }
        self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4, 5), (2, 4, 5, 3), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), (2, 4, 5, 6, 3), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_with_add2(self, input_shape1, input_shape2, perm_input, perm_output):
        node1 = helper.make_node("Transpose", ["input_data1"], ["Y"], perm=perm_input, name="trans")
        node2 = helper.make_node("Add", ["Y", "input_data2"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans2")

        output_shape = input_shape1

        graph = helper.make_graph(
            [node1, node2, node3],
            "transpose_with_shape",
            [helper.make_tensor_value_info("input_data1", TensorProto.FLOAT, input_shape1),
             helper.make_tensor_value_info("input_data2", TensorProto.FLOAT, input_shape2),
             ],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        feed_dict = {"input_data1": np.random.randn(*input_shape1).astype(np.float32),
                     "input_data2": np.random.randn(*input_shape2).astype(np.float32),
                     }
        self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=1)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_relu(self, shape, perm_input, perm_output):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Relu", ["Y"], ["Z"], name="relu")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "relu-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_leaky_relu(self, shape, perm_input, perm_output):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("LeakyRelu", ["Y"], ["Z"], alpha=0.02, name="relu")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "LeakyRelu-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_with_prelu(self, input_shape, perm_input, perm_output):
        node1 = helper.make_node("Transpose", ["input_data1"], ["Y"], perm=perm_input, name="trans")
        node2 = helper.make_node("PRelu", ["Y", "input_data2"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "transpose_with_shape",
            [helper.make_tensor_value_info("input_data1", TensorProto.FLOAT, input_shape),
             helper.make_tensor_value_info("input_data2", TensorProto.FLOAT, (input_shape[1],)),
             ],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        feed_dict = {"input_data1": np.random.randn(*input_shape).astype(np.float32),
                     "input_data2": np.random.randn(input_shape[1]).astype(np.float32),
                     }
        self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(10, "QuantizeLinear")
    def test_transpose_quantize(self, shape, perm_input, perm_output):
        scale = numpy_helper.from_array(np.array(0.75, dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array(3, dtype=np.uint8), name='zero_point')
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("QuantizeLinear", ["Y", "scale", "zero_point"], ["Z"], name="quantize")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "quantize-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.UINT8, shape)],
            [scale, zero_point]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [0, 2, 1], [0, 2, 1]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(13, "QuantizeLinear with axis")
    def test_transpose_quantize_with_axis(self, shape, perm_input, perm_output):
        scale = numpy_helper.from_array(np.array([0.75, 0.1, 2.3, 0.3], dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array([2, 4, 6, 8], dtype=np.uint8), name='zero_point')
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("QuantizeLinear", ["Y", "scale", "zero_point"], ["Z"], name="quantize", axis=1)
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "quantize-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.UINT8, shape)],
            [scale, zero_point]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(10, "DequantizeLinear")
    def test_transpose_dequantize(self, shape, perm_input, perm_output):
        scale = numpy_helper.from_array(np.array(0.75, dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array(3, dtype=np.uint8), name='zero_point')
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("DequantizeLinear", ["Y", "scale", "zero_point"], ["Z"], name="dequantize")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "dequantize-test",
            [helper.make_tensor_value_info("X", TensorProto.UINT8, shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, shape)],
            [scale, zero_point]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randint(0, 100, shape, np.uint8)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [0, 2, 1], [0, 2, 1]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(13, "DequantizeLinear with axis")
    def test_transpose_dequantize_with_axis(self, shape, perm_input, perm_output):
        scale = numpy_helper.from_array(np.array([0.75, 0.1, 2.3, 0.3], dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array([2, 4, 6, 8], dtype=np.uint8), name='zero_point')
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("DequantizeLinear", ["Y", "scale", "zero_point"], ["Z"], name="dequantize", axis=1)
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "dequantize-test",
            [helper.make_tensor_value_info("X", TensorProto.UINT8, shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, shape)],
            [scale, zero_point]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randint(0, 100, shape, np.uint8)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ([2, 3, 4], [1, 2, 1], [1], [0, 2, 1], [0, 2, 1]),
        ([2, 3, 4, 5], [1, 2, 1, 2], [1], [0, 2, 3, 1], [0, 3, 1, 2]),
        ([2, 3, 4, 5], [1, 2, 1, 2], [1, 2], [0, 2, 3, 1], [0, 3, 1, 2]),
        ([2, 3, 4, 5], [1, 2, 1, 2], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]),
        ([2, 3, 4, 5, 6], [1, 2, 1, 2, 1], [2], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ([2, 3, 4, 5, 6], [1, 2, 1, 2, 1], [2, 3], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ([2, 3, 4, 5, 6], [1, 2, 1, 2, 1], [0, 1, 2, 3, 4], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_max_version(9, "Slice in opset 9 and takes 'axes, 'start' and 'ends' as attributes")
    def test_transpose_slice(self, input_shape, slice_size, axes, perm_input, perm_output):
        axes = np.array(axes, dtype=np.int64)
        starts = np.array([0] * axes.size, dtype=np.int64)
        ends = []
        for i in range(axes.size):
            ends.append(slice_size[axes[i]])
        ends = np.array(ends, dtype=np.int64)
        output_shape = input_shape.copy()
        for axis in axes:
            output_shape[perm_input[axis]] = slice_size[axis]
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Slice", ["Y"], ["Z"], starts=starts, ends=ends, axes=axes, name="slice")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "slice-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, output_shape)],
            [
                helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
                helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
                helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes)
            ]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ([2, 3, 4], [1, 2, 1], [1], [0, 2, 1], [0, 2, 1]),
        ([2, 3, 4, 5], [1, 2, 1, 2], [1], [0, 2, 3, 1], [0, 3, 1, 2]),
        ([2, 3, 4, 5], [1, 2, 1, 2], [1, 2], [0, 2, 3, 1], [0, 3, 1, 2]),
        ([2, 3, 4, 5], [1, 2, 1, 2], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]),
        ([2, 3, 4, 5, 6], [1, 2, 1, 2, 1], [2], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ([2, 3, 4, 5, 6], [1, 2, 1, 2, 1], [2, 3], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ([2, 3, 4, 5, 6], [1, 2, 1, 2, 1], [0, 1, 2, 3, 4], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(10, "Slice in opset 10 can accept dynamic 'start' and 'ends'")
    def test_transpose_slice_opset_10(self, input_shape, slice_size, axes, perm_input, perm_output):
        axes = np.array(axes, dtype=np.int32)
        starts = np.array([0] * axes.size, dtype=np.int32)
        ends = []
        for i in range(axes.size):
            ends.append(slice_size[axes[i]])
        ends = np.array(ends, dtype=np.int32)
        output_shape = input_shape.copy()
        for axis in axes:
            output_shape[perm_input[axis]] = slice_size[axis]
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Slice", ["Y", "starts", "ends", "axes"], ["Z"], name="slice")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "slice-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, output_shape)],
            [
                helper.make_tensor("starts", TensorProto.INT32, starts.shape, starts),
                helper.make_tensor("ends", TensorProto.INT32, ends.shape, ends),
                helper.make_tensor("axes", TensorProto.INT32, axes.shape, axes)
            ]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), (4, 2, 3), (2, 0, 1), (1, 2, 0)),
        ((2, 3, 4, 5), (2, 4, 5, 3), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), (2, 4, 5, 6, 3), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(8, "Max in opset 10 supports broadcasting")
    def test_transpose_max(self, input_shape1, input_shape2, perm_input, perm_output):
        const_1_val = [2.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        const_1_node = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")

        const_2_val = np.random.randn(*input_shape2).astype(np.float32)
        const_2 = helper.make_tensor("const_2", TensorProto.FLOAT, input_shape2, const_2_val.flatten())
        const_2_node = helper.make_node("Constant", [], ["const_2"], value=const_2, name="const_2")

        const_3_val = np.random.randn(*input_shape2).astype(np.float32)
        const_3 = helper.make_tensor("const_3", TensorProto.FLOAT, input_shape2, const_3_val.flatten())
        const_3_node = helper.make_node("Constant", [], ["const_3"], value=const_3, name="const_3")

        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Max", ["Y", "const_3", "const_2", "const_1"], ["Z"], name="max")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        output_shape = input_shape1

        graph = helper.make_graph(
            [const_1_node, const_2_node, const_3_node, node1, node2, node3],
            "Max-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape1)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*input_shape1).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4, 5), (2, 4, 5, 3), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), (2, 4, 5, 6, 3), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(8, "Max in opset 10 supports broadcasting")
    def test_transpose_max_input_non_const(self, input_shape1, input_shape2, perm_input, perm_output):
        const_1_val = [2.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        const_1_node = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")

        const_2_val = np.random.randn(*input_shape2).astype(np.float32)
        const_2 = helper.make_tensor("const_2", TensorProto.FLOAT, input_shape2, const_2_val.flatten())
        const_2_node = helper.make_node("Constant", [], ["const_2"], value=const_2, name="const_2")

        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Max", ["Y", "non_const", "const_2", "const_1"], ["Z"], name="max")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=perm_output, name="trans_2")

        output_shape = input_shape1

        graph = helper.make_graph(
            [const_1_node, const_2_node, node1, node2, node3],
            "Max-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape1),
             helper.make_tensor_value_info("non_const", TensorProto.FLOAT, input_shape2)],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(*input_shape1).astype(np.float32),
                                            "non_const": np.random.randn(*input_shape2).astype(np.float32)},
                                   model_proto, remaining_transpose_num=1)

    @parameterized.expand([
        ((2, 3, 4, 5), (2, 4, 5, 3), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), (2, 4, 5, 6, 3), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(8, "Max in opset 10 supports broadcasting")
    def test_transpose_max_no_cancel(self, input_shape1, input_shape2, perm_input, perm_output):
        const_1_val = [2.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        const_1_node = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")

        const_2_val = np.random.randn(*input_shape2).astype(np.float32)
        const_2 = helper.make_tensor("const_2", TensorProto.FLOAT, input_shape2, const_2_val.flatten())
        const_2_node = helper.make_node("Constant", [], ["const_2"], value=const_2, name="const_2")

        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Max", ["Y", "non_const", "const_2", "const_1"], ["Z"], name="max")

        output_shape = [None] * len(input_shape1)

        graph = helper.make_graph(
            [const_1_node, const_2_node, node1, node2],
            "Max-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape1),
             helper.make_tensor_value_info("non_const", TensorProto.FLOAT, input_shape2)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape1).astype(np.float32),
                                           "non_const": np.random.randn(*input_shape2).astype(np.float32)},
                                   model_proto, remaining_transpose_num=2)

    @parameterized.expand([
        ((2, 3, 4, 5), (2, 4, 5, 3), [0, 2, 3, 1]),
        ((2, 3, 4, 5, 6), (2, 4, 5, 6, 3), [0, 2, 3, 4, 1]),
    ])
    def test_transpose_merge(self, input_shape1, input_shape2, perm):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node1 = helper.make_node("Transpose", ["X"], ["Y_1"], perm=perm, name="trans_1")
        node2 = helper.make_node("Mul", ["Y", "Y_1"], ["OUT"], name="mul")

        output_shape = input_shape2

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-merge-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape1)],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(*input_shape1).astype(np.float32)},
                                   model_proto, remaining_transpose_num=1)


    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_mul_as_square(self, shape, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans")
        node1 = helper.make_node("Mul", ["Y", "Y"], ["Z"], name="mul")
        node2 = helper.make_node("Transpose", ["Z"], ["OUT"], perm=perm_output, name="trans_1")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-mul-as-sqr-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_mul_broadcastable_const(self, shape, perm_input, perm_output):
        const = numpy_helper.from_array(np.random.random((1, shape[1])).astype(np.float32), name='const')
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans")
        node1 = helper.make_node("Mul", ["Y", "const"], ["Z"], name="mul")
        node2 = helper.make_node("Transpose", ["Z"], ["OUT"], perm=perm_output, name="trans_1")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-mul-const-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape)],
            [const],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1]),
        ((2, 3, 4, 5), [0, 2, 3, 1]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1]),
    ])
    def test_transpose_with_shape(self, shape, perm):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node2 = helper.make_node("Shape", ["Y"], ["Z"], name="shape")

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_shape",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Z", TensorProto.INT64, [len(shape)])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), (4, 2, 3), [2, 0, 1]),
        ((2, 3, 4, 5), (2, 4, 5, 3), [0, 2, 3, 1]),
        ((2, 3, 4, 5, 6), (2, 4, 5, 6, 3), [0, 2, 3, 4, 1]),
    ])
    def test_transpose_with_identity(self, input_shape, output_shape, perm):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node2 = helper.make_node("Identity", ["Y"], ["Z"], name="identity")

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_identity",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=1)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_sqrt(self, shape, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans1")
        node1 = helper.make_node("Sqrt", ["Y"], ["Z"], name="sqrt")
        node2 = helper.make_node("Transpose", ["Z"], ["OUT"], perm=perm_output, name="trans2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-sqrt-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((1, 3, 4), [4, 3], [0, 2, 1], [1, 0]),
        ((1, 3, 4, 5), (4, 5, 3), [0, 2, 3, 1], [1, 2, 0]),
        ((1, 3, 4, 5, 6), (4, 5, 6, 3), [0, 2, 3, 4, 1], [1, 2, 3, 0]),
    ])
    @check_opset_max_version(12, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze1(self, input_shape, output_shape, perm, expected_perm):
        # squeeze the first dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[0])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        model_after_opt = self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                                     model_proto, remaining_transpose_num=1)
        self.check_transpose_perm(model_after_opt, expected_perm)

    @parameterized.expand([
        ((1, 3, 4), (1, 4, 1, 3, 1, 1), [2, 0, 1], [0, 4, 5], [2, 3, 0, 1, 4, 5]),
        ((1, 3, 4, 5), (1, 1, 4, 5, 1, 3, 1), [0, 2, 3, 1], [0, 4, 6], [0, 1, 4, 5, 2, 3, 6]),
        ((1, 3, 4, 5, 6), (1, 1, 4, 5, 1, 6, 1, 3), [0, 2, 3, 4, 1], [0, 4, 6], [0, 1, 4, 5, 6, 7, 2, 3]),
    ])
    def test_transpose_with_unsqueeze(self, input_shape, output_shape, perm, axes_val, expected_perm):
        # unsqueeze the first dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        if self.config.opset <= 12:
            node2 = helper.make_node("Unsqueeze", ["Y"], ["Z"], name="unsqueeze", axes=axes_val)
            nodes = [node1, node2]
        else:
            axes = self._make_onnx_const(np.array(axes_val, dtype=np.int64), "axes")
            node2 = helper.make_node("Unsqueeze", ["Y", "axes"], ["Z"], name="unsqueeze")
            nodes = [axes, node1, node2]

        graph = helper.make_graph(
            nodes,
            "transpose_with_unsqueeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        model_after_opt = self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                                     model_proto, remaining_transpose_num=1)
        self.check_transpose_perm(model_after_opt, expected_perm)

    @parameterized.expand([
        ((1, 3, 4), [4, 3], [0, 2, 1], [1, 0]),
        ((1, 3, 4, 5), (4, 5, 3), [0, 2, 3, 1], [1, 2, 0]),
        ((1, 3, 4, 5, 6), (4, 5, 6, 3), [0, 2, 3, 4, 1], [1, 2, 3, 0]),
    ])
    @check_opset_min_version(13, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze1_13(self, input_shape, output_shape, perm, expected_perm):
        # squeeze the first dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        axes = self._make_onnx_const(np.array([0], dtype=np.int64), "axes")
        node2 = helper.make_node("Squeeze", ["Y", "axes"], ["Z"], name="squeeze")

        graph = helper.make_graph(
            [node1, node2, axes],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        model_after_opt = self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                                     model_proto, remaining_transpose_num=1)
        self.check_transpose_perm(model_after_opt, expected_perm)

    @parameterized.expand([
        ((3, 4, 1, 5), (3, 5, 4), [0, 2, 3, 1], [0, 2, 1]),
        ((3, 4, 1, 5, 6), (3, 5, 6, 4), [0, 2, 3, 4, 1], [0, 2, 3, 1]),
    ])
    @check_opset_max_version(12, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze2(self, input_shape, output_shape, perm, expected_perm):
        # squeeze the second dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[1])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        model_after_opt = self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                                     model_proto, remaining_transpose_num=1)
        self.check_transpose_perm(model_after_opt, expected_perm)

    @parameterized.expand([
        ((3, 4, 1, 5), (3, 5, 4), [0, 2, 3, 1], [0, 2, 1]),
        ((3, 4, 1, 5, 6), (3, 5, 6, 4), [0, 2, 3, 4, 1], [0, 2, 3, 1]),
    ])
    @check_opset_min_version(13, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze2_13(self, input_shape, output_shape, perm, expected_perm):
        # squeeze the second dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        axes = self._make_onnx_const(np.array([1], dtype=np.int64), "axes")
        node2 = helper.make_node("Squeeze", ["Y", "axes"], ["Z"], name="squeeze")

        graph = helper.make_graph(
            [node1, node2, axes],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        model_after_opt = self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                                     model_proto, remaining_transpose_num=1)
        self.check_transpose_perm(model_after_opt, expected_perm)

    @parameterized.expand([
        ((3, 1, 4, 5), (3, 4, 5), [0, 2, 3, 1]),
        ((3, 1, 4, 5, 6), (3, 4, 5, 6), [0, 2, 3, 4, 1]),
    ])
    @check_opset_max_version(12, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze3(self, input_shape, output_shape, perm):
        # squeeze the last dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[len(input_shape) - 1])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 1, 4, 5), (3, 4, 5), [0, 2, 3, 1]),
        ((3, 1, 4, 5, 6), (3, 4, 5, 6), [0, 2, 3, 4, 1]),
    ])
    @check_opset_min_version(13, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze3_13(self, input_shape, output_shape, perm):
        # squeeze the last dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        axes = self._make_onnx_const(np.array([len(input_shape) - 1], dtype=np.int64), "axes")
        node2 = helper.make_node("Squeeze", ["Y", "axes"], ["Z"], name="squeeze")

        graph = helper.make_graph(
            [node1, node2, axes],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 1, 1, 5), (3, 5), [0, 2, 3, 1]),
        ((3, 1, 1, 5, 4), (3, 5, 4), [0, 2, 3, 4, 1]),
    ])
    @check_opset_max_version(12, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze4(self, input_shape, output_shape, perm):
        # squeeze the two dims
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[1, len(input_shape) - 1])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 1, 1, 5), (3, 5), [0, 2, 3, 1]),
        ((3, 1, 1, 5, 4), (3, 5, 4), [0, 2, 3, 4, 1]),
    ])
    @check_opset_min_version(13, "Squeeze/Unsqueeze changed since opset 13")
    def test_transpose_with_squeeze4_13(self, input_shape, output_shape, perm):
        # squeeze the two dims
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        axes = self._make_onnx_const(np.array([1, len(input_shape) - 1], dtype=np.int64), "axes")
        node2 = helper.make_node("Squeeze", ["Y", "axes"], ["Z"], name="squeeze")

        graph = helper.make_graph(
            [node1, node2, axes],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((10, 3, 4), [0, 2, 1], [0, 2, 1]),
        ((10, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((10, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_with_loop(self, shape, perm_input, perm_output):
        def _define_loop_graph(external_inputs):
            # external_inputs: external node which will be used by this graph
            # graph without loop carried
            # computation
            # for(...){a = external_inputs[i]; b = trans(a), c = squeeze(b)}, c is scan output
            node1 = helper.make_node("Gather", [external_inputs[0], "loop_iter_num"], ["Y0"])
            node2 = helper.make_node("Transpose", ["Y0"], ["Z0"], perm=perm_input)
            # graph output
            if get_test_config().opset <= 12:
                node3 = helper.make_node("Squeeze", ["Z0"], ["scan_output"], axes=[0])
                const_node = None
            else:
                const_tensor = helper.make_tensor(name='const', data_type=TensorProto.INT64, dims=[1],
                                                  vals=np.array([0], dtype=np.int64))
                const_node = helper.make_node("Constant", [], ["axes_const"], value=const_tensor, name="const")
                node3 = helper.make_node("Squeeze", ["Z0", "axes_const"], ["scan_output"])
            node4 = helper.make_node("Identity", ["loop_condition"], ["loop_cond_output"])
            node5 = helper.make_node("Identity", ["loop_condition"], ["loop_carried_output"])

            nodes = [node1, node2, node3, node4, node5]
            if const_node is not None:
                nodes.append(const_node)

            graph = helper.make_graph(
                nodes,
                "loop_subgraph",
                [helper.make_tensor_value_info("loop_iter_num", TensorProto.INT64, (1,)),  # iteration_num
                 helper.make_tensor_value_info("loop_condition", TensorProto.BOOL, ()),  # condition
                 helper.make_tensor_value_info("loop_carried", TensorProto.BOOL, ())  # loop_carried
                 ],
                [helper.make_tensor_value_info("loop_cond_output", TensorProto.BOOL, ()),
                 helper.make_tensor_value_info("loop_carried_output", TensorProto.BOOL, ()),
                 helper.make_tensor_value_info("scan_output", TensorProto.FLOAT, ["unknown"] * (len(shape) - 1))
                 ],
            )
            return graph

        def _make_loop(external_inputs, outputs):
            trip_cnt = self._make_onnx_const(np.array(10, dtype=np.int64), "trip_cnt")
            cond = self._make_onnx_const(np.array(True, dtype=bool), "cond")
            sub_graph = _define_loop_graph(external_inputs)
            loop_node = helper.make_node("Loop", ["trip_cnt", "cond", "cond"], outputs,
                                         name="loop", body=sub_graph)
            return trip_cnt, cond, loop_node

        nodes = _make_loop(["array"], ["loop_carried", "scan_out"])
        res = helper.make_node("Transpose", ["scan_out"], ["Y"], perm=perm_output, name="trans")

        graph = helper.make_graph(
            [*nodes, res],
            "transpose_with_loop",
            [helper.make_tensor_value_info("array", TensorProto.FLOAT, ["unknow"] * len(shape))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["unknow"] * len(shape))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Y"], {"array": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [4, 2, 3], [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [2, 4, 5, 3], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [2, 4, 5, 6, 3], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_trans_with_sub(self, io_shape, const_shape_base, perm_input, perm_output):
        const_shapes = []
        for i in range(len(const_shape_base)):
            const_shapes.append(const_shape_base[i:])
        for trans_is_first_input in [True, False]:
            for const_shape in const_shapes:
                node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_a")
                const_tensor = helper.make_tensor(name='const', data_type=TensorProto.FLOAT, dims=const_shape,
                                                  vals=np.random.randn(*const_shape).flatten().astype(np.float32))
                node2 = helper.make_node("Constant", [], ["const"], value=const_tensor, name="const")
                if trans_is_first_input:
                    node3 = helper.make_node("Sub", ["Y", "const"], ["Z"], name="sub")
                else:
                    node3 = helper.make_node("Sub", ["const", "Y"], ["Z"], name="sub")

                node4 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_b")
                graph = helper.make_graph(
                    [node1, node2, node3, node4],
                    "test_trans_with_sub",
                    [helper.make_tensor_value_info("X", TensorProto.FLOAT, io_shape)],
                    [helper.make_tensor_value_info("res", TensorProto.FLOAT, io_shape)],
                )

                model_proto = self.make_model(graph, producer_name="onnx-tests")
                self.run_transpose_compare(["res"], {"X": np.random.randn(*io_shape).astype(np.float32)},
                                           model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4, 5), [2, 4, 5, 3], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [2, 4, 5, 6, 3], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_trans_with_sub_input_non_const(self, io_shape, non_const_shape_base, perm_input, perm_output):
        non_const_shapes = []
        for i in range(len(non_const_shape_base) - 1):
            non_const_shapes.append(non_const_shape_base[i:])
        for trans_is_first_input in [True, False]:
            for non_const_shape in non_const_shapes:
                node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_a")
                if trans_is_first_input:
                    node2 = helper.make_node("Sub", ["Y", "non_const"], ["Z"], name="sub")
                else:
                    node2 = helper.make_node("Sub", ["non_const", "Y"], ["Z"], name="sub")

                node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_b")
                graph = helper.make_graph(
                    [node1, node2, node3],
                    "test_trans_with_sub_input_non_const",
                    [helper.make_tensor_value_info("X", TensorProto.FLOAT, io_shape),
                     helper.make_tensor_value_info("non_const", TensorProto.FLOAT, non_const_shape)],
                    [helper.make_tensor_value_info("res", TensorProto.FLOAT, io_shape)],
                )

                model_proto = self.make_model(graph, producer_name="onnx-tests")
                self.run_transpose_compare(["res"], {"X": np.random.randn(*io_shape).astype(np.float32),
                                                     "non_const": np.random.randn(*non_const_shape).astype(np.float32)},
                                           model_proto, remaining_transpose_num=1)

    @parameterized.expand([
        ((1, 1, 3, 3), (1, 3, 3, 1), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 1, 3, 3, 3), (1, 3, 3, 3, 1), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_add_with_input_non_const(self, input_shape1, input_shape2, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("Add", ["Y", "A"], ["Z"], name="add")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        output_shape = input_shape1

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-add-test-input-non-const",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape1),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, input_shape2)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape1).astype(np.float32),
                                             "A": np.random.randn(*input_shape2).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [4, 2, 3], [2, 0, 1], [1, 2, 0]),
        ((1, 1, 3, 3), (1, 3, 3, 1), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 1, 3, 3, 3), (1, 3, 3, 3, 1), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_add_with_input_const(self, input_shape1, input_shape2, perm_input, perm_output):
        const_1_val = np.random.randn(*input_shape2).astype(np.float32)
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, input_shape2, const_1_val.flatten())
        const_1_node = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")

        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("Add", ["Y", "const_1"], ["Z"], name="add")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        output_shape = input_shape1

        graph = helper.make_graph(
            [const_1_node, node0, node1, node2],
            "transpose-add-test-input-const",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape1)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape1).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((1, 5, 3, 3), (16, 5, 3, 3), (1, 16, 1, 1), (1, 1, 1, 16), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 5, 3, 3, 3), (16, 5, 3, 3, 3), (1, 16, 1, 1, 1), (1, 1, 1, 1, 16), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_add_with_conv_1(self, input_shape, weights_shape, output_shape,
                                       const_shape, perm_input, perm_output):
        # case where bias's dim is 1D and can be merged into Conv
        const_b_val = np.random.randn(*const_shape).astype(np.float32)
        const_b = helper.make_tensor("const_b", TensorProto.FLOAT, const_shape, const_b_val.flatten())
        const_b_node = helper.make_node("Constant", [], ["const_b"], value=const_b, name="const_b")

        node0 = helper.make_node("Conv", ["x", "W"], ["X"], name="conv", pads=[0] * 2 * (len(input_shape) - 2))
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Add", ["Y", "const_b"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [const_b_node, node0, node1, node2, node3],
            "transpose-add-test-with-conv-1",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
             helper.make_tensor_value_info("W", TensorProto.FLOAT, weights_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"x": np.random.randn(*input_shape).astype(np.float32),
                                             "W": np.random.randn(*weights_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((1, 1, 5, 5), (1, 1, 3, 3), (1, 1, 3, 3), (1, 3, 3, 1), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 1, 5, 5, 5), (1, 1, 3, 3, 3), (1, 1, 3, 3, 3), (1, 3, 3, 3, 1), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_add_with_conv_2(self, input_shape, weights_shape, output_shape,
                                       const_shape, perm_input, perm_output):
        # case where bias's dim is not 1D and can't be merged into Conv
        # add handler just remove the transpose around Add node
        const_b_val = np.random.randn(*const_shape).astype(np.float32)
        const_b = helper.make_tensor("const_b", TensorProto.FLOAT, const_shape, const_b_val.flatten())
        const_b_node = helper.make_node("Constant", [], ["const_b"], value=const_b, name="const_b")

        node0 = helper.make_node("Conv", ["x", "W"], ["X"], name="conv", pads=[0] * 2 * (len(input_shape) - 2))
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node2 = helper.make_node("Add", ["Y", "const_b"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [const_b_node, node0, node1, node2, node3],
            "transpose-add-test-with-conv-2",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
             helper.make_tensor_value_info("W", TensorProto.FLOAT, weights_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"x": np.random.randn(*input_shape).astype(np.float32),
                                             "W": np.random.randn(*weights_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_neg(self, shape, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans1")
        node1 = helper.make_node("Neg", ["Y"], ["Z"], name="neg")
        node2 = helper.make_node("Transpose", ["Z"], ["OUT"], perm=perm_output, name="trans2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-neg-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 4, 5), (8, 4, 6), [1, 3, 0, 0, 2, 0], [2, 0, 1], [1, 2, 0]),
        ((1, 3, 4, 5), (2, 6, 4, 8), [1, 0, 1, 3, 0, 0, 2, 0], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5, 6), (2, 5, 6, 8, 10), [1, 0, 1, 3, 1, 0, 2, 2, 1, 1], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_max_version(10, "pad")
    def test_transpose_pad(self, input_shape, output_shape, pads, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("Pad", ["Y"], ["Z"], pads=pads, name="pad")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-pad-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 4, 5), (8, 4, 6), [1, 3, 0, 0, 2, 0], [2, 0, 1], [1, 2, 0]),
        ((1, 3, 4, 5), (2, 6, 4, 8), [1, 0, 1, 3, 0, 0, 2, 0], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5, 6), (2, 5, 6, 8, 10), [1, 0, 1, 3, 1, 0, 2, 2, 1, 1], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(11, "pad")
    def test_transpose_pad11(self, input_shape, output_shape, pads, perm_input, perm_output):

        pads_val = np.array(pads, dtype=np.int64)
        pads_tensor = helper.make_tensor("Pads", TensorProto.INT64, [len(input_shape) * 2], pads_val)
        pads_const = helper.make_node("Constant", [], ["Pads"], value=pads_tensor, name="Pads")

        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("Pad", ["Y", "Pads"], ["Z"], name="pad")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2, pads_const],
            "transpose-pad-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 4, 5), (8, 4, 6), [1, 3, 0, 0, 2, 0], [2, 0, 1], [1, 2, 0]),
        ((1, 3, 4, 5), (2, 6, 4, 8), [1, 0, 1, 3, 0, 0, 2, 0], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5, 6), (2, 5, 6, 8, 10), [1, 0, 1, 3, 1, 0, 2, 2, 1, 1], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(11, "pad")
    def test_transpose_pad11_non_const_pads(self, input_shape, output_shape, pads, perm_input, perm_output):

        pads_val = np.array(pads, dtype=np.int64)
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("Pad", ["Y", "Pads"], ["Z"], name="pad")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-pad-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
             helper.make_tensor_value_info("Pads", TensorProto.INT64, pads_val.shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"],
                                   {
                                       "X": np.random.randn(*input_shape).astype(np.float32),
                                       "Pads": pads_val
                                   },
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), [2, 0, 1], [1, 2, 0]),
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((2, 3, 4, 5, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_reciprocal(self, shape, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans1")
        node1 = helper.make_node("Reciprocal", ["Y"], ["Z"], name="reciprocal")
        node2 = helper.make_node("Transpose", ["Z"], ["OUT"], perm=perm_output, name="trans2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-reciprocal-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 4, 5), (3, 4, 1), [0, 2, 1], [0, 2, 1]),
        ((1, 3, 4, 5), (1, 3, 1, 1), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5, 6), (1, 3, 1, 1, 1), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_transpose_reducemean(self, input_shape, output_shape, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("ReduceMean", ["Y"], ["Z"], axes=list(range(1, len(input_shape) - 1)),
                                 keepdims=1, name="reducemean")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-reducemean-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 4, 5), (3, 4, 1), [1], [0, 2, 1], [0, 2, 1]),
        ((1, 3, 4, 5), (1, 3, 4, 1), [2], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5), (1, 3, 1, 1), [1, 2], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5), (1, 1, 1, 1), [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5, 6), (1, 3, 1, 5, 6), [1], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ((1, 3, 4, 5, 6), (1, 3, 1, 1, 1), [1, 2, 3], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ((1, 3, 4, 5, 6), (1, 1, 1, 1, 1), [0, 1, 2, 3, 4], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_max_version(12, "ReduceSum from opset <= 12 has axes as an attribute")
    def test_transpose_reducesum(self, input_shape, output_shape, axes, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("ReduceSum", ["Y"], ["Z"], axes=axes,
                                 keepdims=1, name="reducesum")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-reducesum-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((1, 3, 4, 5), (1, 3, 4), [2], [0, 2, 3, 1], [0, 2, 1]),
        ((1, 3, 4, 5), (1, 3), [1, 2], [0, 2, 3, 1], [0, 1]),
        ((1, 3, 4, 5), (), [0, 1, 2, 3], [0, 2, 3, 1], []),
        ((1, 3, 4, 5, 6), (1, 3, 5, 6), [1], [0, 2, 3, 4, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5, 6), (1, 3), [1, 2, 3], [0, 2, 3, 4, 1], [0, 1]),
        ((1, 3, 4, 5, 6), (), [0, 1, 2, 3, 4], [0, 2, 3, 4, 1], []),
    ])
    def test_transpose_reducemax(self, input_shape, output_shape, axes, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("ReduceMax", ["Y"], ["Z"], axes=axes,
                                 keepdims=0, name="reducemax")
        if perm_output:
            node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")
        else:
            node2 = helper.make_node("Identity", ["Z"], ["res"], name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-reducemax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_transpose_argmax(self):
        input_shape = [1, 2, 3, 4]
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("ArgMax", ["Y"], ["Z"], axis=3, keepdims=0, name="argmax")
        node2 = helper.make_node("Cast", ["Z"], ["res"], to=TensorProto.INT32, name="cast")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-argmax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.INT32, [1, 3, 4])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @check_opset_max_version(
        12, "Before opset 13, Softmax coerced its inputs to 2D and can thus only be optimized for certain permutations"
    )
    def test_transpose_softmax_valid_perm(self):
        input_shape = [4, 4, 4, 4]
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("Softmax", ["Y"], ["Z"], axis=1, name="softmax")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-softmax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(
            ["res"], {"X": np.random.randn(*input_shape).astype(np.float32)}, model_proto, remaining_transpose_num=0
        )

    @check_opset_max_version(
        12, "Before opset 13, Softmax coerced its inputs to 2D and can thus only be optimized for certain permutations"
    )
    def test_transpose_softmax_invalid_perm(self):
        input_shape = [4, 4, 4, 4]
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("Softmax", ["Y"], ["Z"], axis=3, name="softmax")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-softmax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(
            ["res"], {"X": np.random.randn(*input_shape).astype(np.float32)}, model_proto, remaining_transpose_num=2
        )

    @check_opset_min_version(13, "Softmax can be optimized for all permutations since opset 13")
    def test_transpose_softmax_13(self):
        input_shape = [4, 4, 4, 4]
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("Softmax", ["Y"], ["Z"], axis=3, name="softmax")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-softmax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(
            ["res"], {"X": np.random.randn(*input_shape).astype(np.float32)}, model_proto, remaining_transpose_num=0
        )

    @check_opset_max_version(
        12,
        "Before opset 13, LogSoftmax coerced its inputs to 2D and can thus only be optimized for certain permutations",
    )
    def test_transpose_logsoftmax_valid_perm(self):
        input_shape = [4, 4, 4, 4]
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("LogSoftmax", ["Y"], ["Z"], axis=1, name="logsoftmax")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-logsoftmax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(
            ["res"], {"X": np.random.randn(*input_shape).astype(np.float32)}, model_proto, remaining_transpose_num=0
        )

    @check_opset_max_version(
        12,
        "Before opset 13, LogSoftmax coerced its inputs to 2D and can thus only be optimized for certain permutations",
    )
    def test_transpose_logsoftmax_invalid_perm(self):
        input_shape = [4, 4, 4, 4]
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("LogSoftmax", ["Y"], ["Z"], axis=3, name="logsoftmax")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-logsoftmax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(
            ["res"], {"X": np.random.randn(*input_shape).astype(np.float32)}, model_proto, remaining_transpose_num=2
        )

    @check_opset_min_version(13, "LogSoftmax can be optimized for all permutations since opset 13")
    def test_transpose_logsoftmax_13(self):
        input_shape = [4, 4, 4, 4]
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("LogSoftmax", ["Y"], ["Z"], axis=3, name="logsoftmax")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-logsoftmax-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, input_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(
            ["res"], {"X": np.random.randn(*input_shape).astype(np.float32)}, model_proto, remaining_transpose_num=0
        )

    def test_transpose_tile(self):
        input_shape = [1, 2, 3, 4]

        repeats_value = [3, 6, 5, 11]
        repeats_tensor = helper.make_tensor("A", TensorProto.INT64, [len(input_shape)], repeats_value)
        repeats_const = helper.make_node("Constant", [], ["A"], value=repeats_tensor, name="repeats_const")
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node1 = helper.make_node("Tile", ["Y", "A"], ["Z"], name="tile")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [repeats_const, node0, node1, node2],
            "transpose-tile-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, [3, 22, 18, 20])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((3, 4, 5), (3, 4, 1), [1], [0, 2, 1], [0, 2, 1]),
        ((1, 3, 4, 5), (1, 3, 4, 1), [2], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5), (1, 3, 1, 1), [1, 2], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5), (1, 1, 1, 1), [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 3, 4, 5, 6), (1, 3, 1, 5, 6), [1], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ((1, 3, 4, 5, 6), (1, 3, 1, 1, 1), [1, 2, 3], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
        ((1, 3, 4, 5, 6), (1, 1, 1, 1, 1), [0, 1, 2, 3, 4], [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    @check_opset_min_version(13, "ReduceSum from opset >= 13 has axes as an input")
    def test_transpose_reducesum_opset_13(self, input_shape, output_shape, axes, perm_input, perm_output):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm_input, name="trans_1")
        node1 = helper.make_node("ReduceSum", ["Y", "axes"], ["Z"], keepdims=1, name="reducesum")
        node2 = helper.make_node("Transpose", ["Z"], ["res"], perm=perm_output, name="trans_2")

        axes = np.array(axes, dtype=np.int64)

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-reducesum-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
            [helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*input_shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 3, 4), (4, 2, 3), [2, 0, 1]),
        ((2, 3, 4, 5), (2, 4, 5, 3), [0, 2, 3, 1]),
        ((2, 3, 4, 5, 6), (2, 4, 5, 6, 3), [0, 2, 3, 4, 1]),
    ])
    def test_trans_output_as_graph_outputs(self, input_shape, output_shape, perm):
        """
        If transpose's output is graph's output, don't optimize it.
        """
        trans = helper.make_node("Transpose", ["X"], ["Y"], name="trans", perm=perm)
        graph_proto = helper.make_graph(
            [trans],
            "trans-to-graph-output",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
        )

        graph = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        # remove identity to graph output
        identity_op = graph.get_node_by_output(graph.outputs[0])
        graph.outputs = [identity_op.input[0]]
        graph.remove_node(identity_op.name)

        optimized_graph = GraphUtil.optimize_graph(graph)

        self.assertTrue(optimized_graph, msg="graph after optimizer should not be None")

        trans_cnt = len(group_nodes_by_type(optimized_graph)["Transpose"])

        self.assertTrue(trans_cnt == 1, msg="Expect 1 Transpose ops left, but actually " + str(trans_cnt) + " left")

    @parameterized.expand([
        ((2, 3, 4, 1), (2, 3, 4, 1), [0, 3, 1, 2]),
        ((2, 1, 1, 4), (2, 1, 1, 4), [0, 3, 1, 2]),
        ((2, 3, 4, 1), (2, -1, -1, 1), [0, 3, 1, 2]),
        ((2, 3, 4, 2, 1), (2, 3, 4, 2, 1), [0, 4, 1, 2, 3]),
        ((2, 1, 1, 1, 4), (2, 1, 1, 1, 4), [0, 4, 1, 2, 3]),
        ((2, 3, 4, 2, 1), (2, -1, -1, -1, 1), [0, 4, 1, 2, 3]),
    ])
    def test_trans_can_be_replaced_with_reshape1(self, input_shape_np, input_shape, perm):
        # test trans-NHWC
        result_shape = [input_shape[i] for i in perm]
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        graph = helper.make_graph(
            [node1],
            "test_trans_can_be_replaced_with_reshape",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, result_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Y"], {"X": np.random.randn(*input_shape_np).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((2, 1, 3, 4), (2, 1, 3, 4), [0, 2, 3, 1]),
        ((2, 4, 1, 1), (2, 4, 1, 1), [0, 2, 3, 1]),
        ((2, 1, 3, 4), (2, 1, -1, -1), [0, 2, 3, 1]),
        ((2, 1, 3, 4, 2), (2, 1, 3, 4, 2), [0, 2, 3, 4, 1]),
        ((2, 4, 1, 1, 1), (2, 4, 1, 1, 1), [0, 2, 3, 4, 1]),
        ((2, 1, 3, 4, 2), (2, 1, -1, -1, -1), [0, 2, 3, 4, 1]),
    ])
    def test_trans_can_be_replaced_with_reshape2(self, input_shape_np, input_shape, perm):
        # test trans-NCHW
        result_shape = [input_shape[i] for i in perm]
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
        graph = helper.make_graph(
            [node1],
            "test_trans_can_be_replaced_with_reshape",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, result_shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Y"], {"X": np.random.randn(*input_shape_np).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((1, 6, 8), [2, 0, 1], [1, 2, 0]),
        ((1, 6, 8, 9), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 6, 8, 9, 2), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_two_transposes_switch_with_mul(self, shape, perm_input, perm_output):
        const_node = self._make_onnx_const(np.array(np.random.random(6), dtype=np.float32), "const_10")
        node0 = helper.make_node("Transpose", ["u1"], ["v1"], perm=perm_input, name="trans_0")
        node1 = helper.make_node("Transpose", ["u2"], ["v2"], perm=perm_input, name="trans_1")

        node2 = helper.make_node("Mul", ["v1", "v2"], ["x"], name="mul_1")
        node3 = helper.make_node("Mul", ["x", const_node.output[0]], ["y"], name="mul_2")
        node4 = helper.make_node("Transpose", ["y"], ["res"], perm=perm_output, name="trans_3")

        graph = helper.make_graph(
            [const_node, node0, node1, node2, node3, node4],
            "test-transpose-mul",
            [helper.make_tensor_value_info("u1", TensorProto.FLOAT, shape),
             helper.make_tensor_value_info("u2", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"u1": np.random.randn(*shape).astype(np.float32),
                                             "u2": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    @parameterized.expand([
        ((1, 6, 8, 9), (1, 8, 9, 6), [0, 2, 3, 1], [0, 3, 1, 2]),
        ((1, 6, 8, 9, 2), (1, 8, 9, 2, 6), [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]),
    ])
    def test_many_transposes_and_constant_switch_with_sum(self, input_shape1, input_shape2, perm_input, perm_output):
        constnode = self._make_onnx_const(np.array(np.random.random(input_shape2), dtype=np.float32), "v4")
        node0 = helper.make_node("Transpose", ["u1"], ["v1"], perm=perm_input, name="trans_0")
        node1 = helper.make_node("Transpose", ["u2"], ["v2"], perm=perm_input, name="trans_1")
        node11 = helper.make_node("Transpose", ["u3"], ["v3"], perm=perm_input, name="trans_2")

        node2 = helper.make_node("Sum", ["v1", "v2", "v3", "v4"], ["x"], name="sum_1")
        node3 = helper.make_node("Sum", ["x", "v1"], ["y"], name="sum_2")
        node4 = helper.make_node("Transpose", ["y"], ["res"], perm=perm_output, name="trans_4")

        output_shape = input_shape1

        graph = helper.make_graph(
            [constnode, node0, node1, node11, node2, node3, node4],
            "test-transpose-mul",
            [helper.make_tensor_value_info("u1", TensorProto.FLOAT, input_shape1),
             helper.make_tensor_value_info("u2", TensorProto.FLOAT, input_shape1),
             helper.make_tensor_value_info("u3", TensorProto.FLOAT, input_shape1)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
        )
        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"u1": np.random.randn(*input_shape1).astype(np.float32),
                                             "u2": np.random.randn(*input_shape1).astype(np.float32),
                                             "u3": np.random.randn(*input_shape1).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    # Tranpose Optimizer Tests End

    # Identity Optimizer Tests Start

    def run_identity_compare(self, output_names_with_port, onnx_feed_dict, origin_proto,
                             remaining_identity_num=None, debug=False, rtol=1e-07):
        self.run_and_compare(output_names_with_port, onnx_feed_dict, origin_proto, op_type="Identity",
                             remaining_op_num=remaining_identity_num, debug=debug, rtol=rtol)

    def test_identity_non_graph_output(self):
        node1 = helper.make_node("Add", ["X", "X"], ["Y"], name="add")
        node2 = helper.make_node("Identity", ["Y"], ["Z"], name="identity")
        node3 = helper.make_node("Shape", ["Z"], ["Z1"], name="shape")

        graph = helper.make_graph(
            [node1, node2, node3],
            "identity-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z1", TensorProto.INT64, [4])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_identity_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                  model_proto, remaining_identity_num=0)

    def test_identity_unremovable_identity(self):
        # should not remove!!
        node1 = helper.make_node("Identity", ["X"], ["Y"], name="identity")

        graph = helper.make_graph(
            [node1],
            "identity-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_identity_compare(["Y"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                  model_proto, remaining_identity_num=1)

    def test_identity_output_as_multiple_graph_outputs(self):
        # handle case like this, both Identity nodes are graph outputs,
        #            Add
        #           /   \
        #    Identity   Identity
        # We at most can remove one Identity for this case.
        node1 = helper.make_node("Add", ["X", "X"], ["Y"], name="identity")
        node2 = helper.make_node("Identity", ["Y"], ["Z1"], name="identity2")
        node3 = helper.make_node("Identity", ["Y"], ["Z2"], name="identity3")
        graph = helper.make_graph(
            [node1, node2, node3],
            "identity-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (2, 3, 4, 5)),
             helper.make_tensor_value_info("Z2", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_identity_compare(["Z1", "Z2"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                  model_proto, remaining_identity_num=1)

    def test_identity_in_subgraph_non_graph_output(self):
        node1 = helper.make_node("Add", ["X", "X"], ["Y"], name="add")

        iter_num_value = np.array(1, dtype=np.int64)
        node2 = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['iterate_num_value'],
            value=helper.make_tensor(
                name='iterate_num_value',
                data_type=TensorProto.INT64,
                dims=iter_num_value.shape,
                vals=iter_num_value.flatten().astype(np.int64).tolist(),
            ),
        )

        cond_value = np.array(True, dtype=bool)
        node3 = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['cond_value'],
            value=helper.make_tensor(
                name='cond_value',
                data_type=TensorProto.BOOL,
                dims=iter_num_value.shape,
                vals=cond_value.flatten().astype(bool).tolist(),
            ),
        )

        # sub graph
        sub_node1 = helper.make_node("Add", ["loop_var_1", "loop_var_1"], ["SubY"], name="sub_add")
        sub_node2 = helper.make_node("Identity", ["SubY"], ["SubIdentity1"], name="sub_identity_1")
        sub_node3 = helper.make_node("Identity", ["SubIdentity1"], ["loop_var_out_1"], name="sub_identity_2")
        sub_node4 = helper.make_node("Identity", ["loop_condition"], ["loop_cond_output"], name="sub_identity_3")
        sub_graph = helper.make_graph(
            [sub_node1, sub_node2, sub_node3, sub_node4],
            "identity_subgraph-test",
            [helper.make_tensor_value_info("loop_iter_num", TensorProto.INT64, (1,)),  # iteration_num
             helper.make_tensor_value_info("loop_condition", TensorProto.BOOL, ()),  # condition
             helper.make_tensor_value_info("loop_var_1", TensorProto.FLOAT, ()),  # loop-carried dependency
             ],
            [helper.make_tensor_value_info("loop_cond_output", TensorProto.BOOL, ()),
             helper.make_tensor_value_info("loop_var_out_1", TensorProto.FLOAT, ())
             ],
        )
        # sub graph ends

        loop_node = helper.make_node("Loop", ["iterate_num_value", "cond_value", "Y"], ["loop_var_1_output"],
                                     name="loop", body=sub_graph)

        node4 = helper.make_node("Identity", ["loop_var_1_output"], ["Z"], name="identity")
        node5 = helper.make_node("Shape", ["Z"], ["Z1"], name="shape")

        graph = helper.make_graph(
            [node1, node2, node3, loop_node, node4, node5],
            "identity-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z1", TensorProto.INT64, [4])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_identity_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                  model_proto, remaining_identity_num=0)

    # Identity Optimizer Tests End

    # Merge Duplicated Nodes Optimizer Tests Start

    def run_merge_duplicated_nodes_compare(self, output_names_with_port, onnx_feed_dict, origin_proto,
                                           op_type=None, remaining_op_num=None, debug=False, rtol=1e-07,
                                           graph_validator=None):
        new_proto = self.run_and_compare(output_names_with_port, onnx_feed_dict, origin_proto, op_type=op_type,
                                         remaining_op_num=remaining_op_num, debug=debug, rtol=rtol)
        if graph_validator:
            self.assertTrue(graph_validator(new_proto.graph))

    def test_duplicated_duplicated_input(self):
        # same input or not
        node0 = helper.make_node('Add', inputs=["X", "X"], outputs=["value0"])
        node1 = helper.make_node('Add', inputs=["X", "X"], outputs=["value1"])
        node2 = helper.make_node('Add', inputs=["value1", "X"], outputs=["value2"])
        node3 = helper.make_node("Mul", ["value0", "value2"], ["value3"])
        node4 = helper.make_node("Mul", ["value1", "value3"], ["OUT"])

        graph = helper.make_graph(
            [node0, node1, node2, node3, node4],
            "test_duplicated_duplicated_input",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5))],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, (5, 5))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["OUT"], {"X": np.random.randn(5, 5).astype(np.float32)}, model_proto,
                                                op_type="Add", remaining_op_num=2)

    def test_duplicated_duplicated_attributes(self):
        # same attr or not
        node0 = helper.make_node('ReduceMin', inputs=["X"], outputs=["value0"], axes=[0], keepdims=0)
        node1 = helper.make_node('ReduceMin', inputs=["X"], outputs=["value1"], axes=[0], keepdims=0)
        node2 = helper.make_node('ReduceMin', inputs=["X"], outputs=["value2"], axes=[1], keepdims=0)
        node3 = helper.make_node('Add', inputs=["value0", "value1"], outputs=["value3"])
        node4 = helper.make_node("Mul", ["value2", "value3"], ["OUT"])

        graph = helper.make_graph(
            [node0, node1, node2, node3, node4],
            "test_duplicated_duplicated_attributes",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5))],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, (5,))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["OUT"], {"X": np.random.randn(5, 5).astype(np.float32)}, model_proto,
                                                op_type="ReduceMin", remaining_op_num=2)

    def _check_initializer_num(self, graph_proto, num):
        return num == len(graph_proto.initializer)

    def test_duplicated_duplicated_constant(self):
        const_val = np.array([1, 2, 3], dtype=np.float32)
        tensor_1 = helper.make_tensor("tensor_1", TensorProto.FLOAT, const_val.shape, const_val)
        tensor_2 = helper.make_tensor("tensor_2", TensorProto.FLOAT, const_val.shape, const_val)
        tensor_3 = helper.make_tensor("tensor_3", TensorProto.FLOAT, const_val.shape, const_val)
        tensor_4 = helper.make_tensor("tensor_4", TensorProto.FLOAT, const_val.shape, const_val)
        node0 = helper.make_node('Constant', inputs=[], outputs=["value0"], value=tensor_1)
        node1 = helper.make_node('Constant', inputs=[], outputs=["value1"], value=tensor_2)
        node2 = helper.make_node('Constant', inputs=[], outputs=["value2"], value=tensor_3)
        node3 = helper.make_node('Constant', inputs=[], outputs=["value3"], value=tensor_4)
        node4 = helper.make_node("Mul", ["value0", "value1"], ["output1"])
        node5 = helper.make_node("Mul", ["value2", "output1"], ["output2"])
        node6 = helper.make_node("Mul", ["value3", "output2"], ["OUT"])

        graph = helper.make_graph(
            [node0, node1, node2, node3, node4, node5, node6],
            "test_duplicated_duplicated_constant",
            [],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, (3,))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["OUT"], {}, model_proto, op_type="Constant", remaining_op_num=0,
                                                graph_validator=lambda g: self._check_initializer_num(g, 1))

    def test_duplicated_duplicated_constant_and_initializer(self):
        const_val = np.array([1, 2, 3], dtype=np.float32)
        tensor_1 = helper.make_tensor("value0", TensorProto.FLOAT, const_val.shape, const_val.tobytes(), raw=True)
        tensor_2 = helper.make_tensor("value1", TensorProto.FLOAT, const_val.shape, const_val.tobytes(), raw=True)
        tensor_3 = helper.make_tensor("value2", TensorProto.FLOAT, const_val.shape, const_val.tobytes(), raw=True)
        tensor_4 = helper.make_tensor("value3", TensorProto.FLOAT, const_val.shape, const_val.tobytes(), raw=True)
        node0 = helper.make_node('Constant', inputs=[], outputs=["value0"], value=tensor_1)
        node1 = helper.make_node('Constant', inputs=[], outputs=["value1"], value=tensor_2)
        node4 = helper.make_node("Mul", ["value0", "value1"], ["output1"])
        node5 = helper.make_node("Mul", ["value2", "output1"], ["output2"])
        node6 = helper.make_node("Mul", ["value3", "output2"], ["OUT"])

        graph = helper.make_graph(
            [node0, node1, node4, node5, node6],
            "test_duplicated_duplicated_constant",
            [helper.make_tensor_value_info("value2", TensorProto.FLOAT, (3,))],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, (3,))],
            [tensor_3, tensor_4]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["OUT"], {}, model_proto, op_type="Constant", remaining_op_num=0,
                                                graph_validator=lambda g: self._check_initializer_num(g, 1))

    def test_duplicated_node_is_graph_output(self):
        node0 = helper.make_node('Add', inputs=["X", "X"], outputs=["value0"])
        node1 = helper.make_node('Add', inputs=["X", "X"], outputs=["value1"])
        node2 = helper.make_node('Add', inputs=["value1", "X"], outputs=["value2"])

        graph = helper.make_graph(
            [node0, node1, node2],
            "test_duplicated_node_is_graph_output",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5))],
            [helper.make_tensor_value_info("value1", TensorProto.FLOAT, (5, 5)),
             helper.make_tensor_value_info("value2", TensorProto.FLOAT, (5, 5))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["value1", "value2"],
                                                {"X": np.random.randn(5, 5).astype(np.float32)}, model_proto,
                                                op_type="Add", remaining_op_num=2)

    @check_opset_min_version(10, "Dropout in opset 10 produces mask of 'bool' type")
    def test_duplicated_different_output_length(self):
        node0 = helper.make_node('Dropout', inputs=["X"], outputs=["value0"])
        node1 = helper.make_node('Dropout', inputs=["X"], outputs=["value1", "mask"])
        node2 = helper.make_node('Dropout', inputs=["value1"], outputs=["value2"])

        graph = helper.make_graph(
            [node0, node1, node2],
            "test_duplicated_different_output_length",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("value1", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("mask", TensorProto.BOOL, (5,)),
             helper.make_tensor_value_info("value2", TensorProto.FLOAT, (5,))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["value1", "mask", "value2"],
                                                {"X": np.random.randn(5).astype(np.float32)},
                                                model_proto,
                                                op_type="Dropout", remaining_op_num=2)

    def test_duplicated_need_multiple_run(self):
        node00 = helper.make_node('Log', inputs=["X"], outputs=["value00"])
        node01 = helper.make_node('Log', inputs=["value00"], outputs=["value01"])
        node02 = helper.make_node('Log', inputs=["value01"], outputs=["value02"])

        node10 = helper.make_node('Log', inputs=["X"], outputs=["value10"])
        node11 = helper.make_node('Log', inputs=["value10"], outputs=["value11"])
        node12 = helper.make_node('Log', inputs=["value11"], outputs=["value12"])

        res = helper.make_node('Add', inputs=["value02", "value12"], outputs=["res"])

        graph = helper.make_graph(
            [node00, node01, node02, node10, node11, node12, res],
            "test_duplicated_node_is_graph_output",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, (5,))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["res"], {"X": np.random.randn(5).astype(np.float32)},
                                                model_proto,
                                                op_type="Log", remaining_op_num=3)

    # Merge Duplicated Nodes Optimizer Tests End

    # Reshape Optimizer Tests Start

    @parameterized.expand([
        (["dims12", "dim0_unsq"], 0, 1, 3),  # Reshape [3, 7, 11] -> [7, 11, 3]
        (["dim0_unsq", "dims12"], 2, 0, 2),  # Reshape [3, 7, 11] -> [11, 3, 7]
    ])
    def test_reshape_opt(self, concat_order, gather_i, starts, ends):
        x_shape = [3, 7, 11]
        node0 = helper.make_node("Shape", ["X"], ["S"])
        g_indices_tensor = helper.make_tensor(name='g_indices_tensor', data_type=TensorProto.INT64, dims=[],
                                              vals=np.array([gather_i], np.int64))
        starts_tensor = helper.make_tensor(name='starts_tensor', data_type=TensorProto.INT64, dims=[1],
                                           vals=np.array([starts], np.int64))
        ends_tensor = helper.make_tensor(name='ends_tensor', data_type=TensorProto.INT64, dims=[1],
                                         vals=np.array([ends], np.int64))
        axes_tensor = helper.make_tensor(name='axes_tensor', data_type=TensorProto.INT64, dims=[1],
                                         vals=np.array([0], np.int64))
        node1 = helper.make_node("Constant", [], ["g_indices"], value=g_indices_tensor)
        node2 = helper.make_node("Constant", [], ["starts"], value=starts_tensor)
        node3 = helper.make_node("Constant", [], ["ends"], value=ends_tensor)
        node4 = helper.make_node("Constant", [], ["axes"], value=axes_tensor)
        node5 = helper.make_node("Gather", ["S", "g_indices"], ["dim0"])
        if self.config.opset >= 10:
            node6 = helper.make_node("Slice", ["S", "starts", "ends", "axes"], ["dims12"])
        else:
            node6 = helper.make_node("Slice", ["S"], ["dims12"], starts=[starts], ends=[ends], axes=[0])
        if self.config.opset >= 13:
            node7 = helper.make_node("Unsqueeze", ["dim0", "axes"], ["dim0_unsq"])
        else:
            node7 = helper.make_node("Unsqueeze", ["dim0"], ["dim0_unsq"], axes=[0])
        node8 = helper.make_node("Concat", concat_order, ["dims120"], axis=0)
        node9 = helper.make_node("Reshape", ["X", "dims120"], ["Y"])

        graph = helper.make_graph(
            [node0, node1, node2, node3, node4, node5, node6, node7, node8, node9],
            "test_reshape_opt1",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["Y"], {"X": np.random.randn(*x_shape).astype(np.float32)},
                             model_proto, op_type="Shape", remaining_op_num=0)


    def test_reshape_opt_with_mul(self):
        x_shape = [7, 10, 20, 30]
        node0 = helper.make_node("Shape", ["X"], ["S"])

        g_indices_tensor = helper.make_tensor(name='g_indices_tensor', data_type=TensorProto.INT64, dims=[2],
                                              vals=np.array([1, 2], np.int64))
        starts_tensor = helper.make_tensor(name='starts_tensor', data_type=TensorProto.INT64, dims=[1],
                                           vals=np.array([0], np.int64))
        ends_tensor = helper.make_tensor(name='ends_tensor', data_type=TensorProto.INT64, dims=[1],
                                         vals=np.array([1], np.int64))
        axes_tensor = helper.make_tensor(name='axes_tensor', data_type=TensorProto.INT64, dims=[1],
                                         vals=np.array([0], np.int64))
        five_tensor = helper.make_tensor(name='five_tensor', data_type=TensorProto.INT32, dims=[],
                                         vals=np.array([5], np.int32))
        six_tensor = helper.make_tensor(name='six_tensor', data_type=TensorProto.INT64, dims=[1],
                                        vals=np.array([6], np.int64))
        node1 = helper.make_node("Constant", [], ["g_indices"], value=g_indices_tensor)
        node2 = helper.make_node("Constant", [], ["starts"], value=starts_tensor)
        node3 = helper.make_node("Constant", [], ["ends"], value=ends_tensor)
        node4 = helper.make_node("Constant", [], ["axes"], value=axes_tensor)
        node5 = helper.make_node("Constant", [], ["five"], value=five_tensor)
        node55 = helper.make_node("Constant", [], ["six"], value=six_tensor)

        node6 = helper.make_node("Gather", ["S", "g_indices"], ["dims12"])
        node7 = helper.make_node("ReduceProd", ["dims12"], ["dims12_prod"], axes=[0])
        if self.config.opset >= 10:
            node8 = helper.make_node("Slice", ["S", "starts", "ends", ""], ["dim0"])
        else:
            node8 = helper.make_node("Slice", ["S"], ["dim0"], starts=[0], ends=[1])
        node9 = helper.make_node("Cast", ["dim0"], ["dim0_cast"], to=TensorProto.INT32)

        if self.config.opset >= 13:
            node10 = helper.make_node("Squeeze", ["dim0_cast", "axes"], ["dim0_sq"])
        else:
            node10 = helper.make_node("Squeeze", ["dim0_cast"], ["dim0_sq"], axes=[0])
        node11 = helper.make_node("Mul", ["dim0_sq", "five"], ["five_dim0"])
        if self.config.opset >= 13:
            node12 = helper.make_node("Unsqueeze", ["five_dim0", "axes"], ["five_dim0_unsq"])
        else:
            node12 = helper.make_node("Unsqueeze", ["five_dim0"], ["five_dim0_unsq"], axes=[0])
        node13 = helper.make_node("Cast", ["five_dim0_unsq"], ["five_dim0_cast"], to=TensorProto.INT64)

        node14 = helper.make_node("Concat", ["five_dim0_cast", "dims12_prod", "six"], ["shape"], axis=0)
        node15 = helper.make_node("Reshape", ["X", "shape"], ["Y"])

        graph = helper.make_graph(
            [node0, node1, node2, node3, node4, node5, node55, node6, node7, node8, node9, node10,
             node11, node12, node13, node14, node15],
            "test_reshape_opt1",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 10, 20, 30])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["Y"], {"X": np.random.randn(*x_shape).astype(np.float32)},
                             model_proto, op_type="Shape", remaining_op_num=0)

    # Reshape Optimizer Tests End

    # Const Fold Optimizer Tests Start

    def test_const_fold_trans_with_const1(self):
        shape = (6, 6)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node2 = helper.make_node("Transpose", ["const"], ["value1"])
        node3 = helper.make_node("Add", ["value1", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3],
            "test_const_fold_trans_with_const1",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_const_fold_trans_with_const2(self):
        # need multiple optimization run
        shape = (6, 6)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node2 = helper.make_node("Transpose", ["const"], ["value1"])
        node3 = helper.make_node("Transpose", ["value1"], ["value2"])
        node4 = helper.make_node("Add", ["value2", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3, node4],
            "test_const_fold_trans_with_const2",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"X": np.random.randn(*shape).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_const_fold_node_is_output(self):
        # need multiple optimization run
        shape = (6, 6)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node2 = helper.make_node("Transpose", ["const"], ["value1"])
        node3 = helper.make_node("Transpose", ["value1"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3],
            "test_const_fold_node_is_output",
            [],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {},
                                   model_proto, remaining_transpose_num=0)

    def test_const_fold_concat(self):
        shape = (6, 4)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(*shape).flatten().astype(np.float32))
        const_tensor2 = helper.make_tensor(name='const_tensor2', data_type=TensorProto.FLOAT, dims=shape,
                                           vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node2 = helper.make_node("Constant", [], ["const2"], value=const_tensor2)
        node3 = helper.make_node("Concat", ["const", "const2", "const"], ["value1"], axis=1)
        node4 = helper.make_node("Add", ["value1", "inp"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3, node4],
            "test_const_fold_trans_with_const2",
            [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [6, 12])],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, [6, 12])],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"inp": np.random.randn(6, 12).astype(np.float32)}, model_proto,
                             "Concat", 0)

    @check_opset_max_version(12, "Squeeze/Unsqueeze changed since opset 13")
    def test_const_fold_unsqueeze_with_const(self):
        shape = (6, 6)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node2 = helper.make_node("Unsqueeze", ["const"], ["value1"], axes=[0, 2, 3])
        node3 = helper.make_node("Add", ["value1", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3],
            "test_const_fold_unsqueeze_with_const",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, (1, 6, 1, 1, 6))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"X": np.random.randn(1).astype(np.float32)}, model_proto,
                             "Unsqueeze", 0)

    @check_opset_min_version(13, "Squeeze/Unsqueeze changed since opset 13")
    def test_const_fold_unsqueeze_with_const_13(self):
        shape = (6, 6)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        axes = self._make_onnx_const(np.array([0, 2, 3], dtype=np.int64), "axes")
        node2 = helper.make_node("Unsqueeze", ["const", "axes"], ["value1"])
        node3 = helper.make_node("Add", ["value1", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3, axes],
            "test_const_fold_unsqueeze_with_const",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, (1, 6, 1, 1, 6))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"X": np.random.randn(1).astype(np.float32)}, model_proto,
                             "Unsqueeze", 0)

    def test_const_fold_cast_with_const(self):
        shape = (6, 6)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node2 = helper.make_node("Cast", ["const"], ["value1"], to=TensorProto.INT64)
        node3 = helper.make_node("Add", ["value1", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3],
            "test_const_fold_cast_with_const",
            [helper.make_tensor_value_info("X", TensorProto.INT64, shape)],
            [helper.make_tensor_value_info("res", TensorProto.INT64, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"X": np.random.randn(*shape).astype(np.int64)}, model_proto,
                             "Cast", 0)

    def test_const_fold_add(self):
        shape = (6, 6)
        const_tensor1 = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                           vals=np.random.randn(*shape).flatten().astype(np.float32))
        const_tensor2 = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                           vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const1"], value=const_tensor1)
        node2 = helper.make_node("Constant", [], ["const2"], value=const_tensor2)
        node3 = helper.make_node("Add", ["const1", "const2"], ["add"])
        node4 = helper.make_node("Add", ["add", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3, node4],
            "test_const_fold_add",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"X": np.random.randn(*shape).astype(np.float32)}, model_proto,
                             "Add", 1)

    def test_const_fold_sub(self):
        shape = (6, 6)
        const_tensor1 = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                           vals=np.random.randn(*shape).flatten().astype(np.float32))
        const_tensor2 = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                           vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const1"], value=const_tensor1)
        node2 = helper.make_node("Constant", [], ["const2"], value=const_tensor2)
        node3 = helper.make_node("Sub", ["const1", "const2"], ["sub"])
        node4 = helper.make_node("Sub", ["sub", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3, node4],
            "test_const_fold_sub",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"X": np.random.randn(*shape).astype(np.float32)}, model_proto,
                             "Sub", 1)

    def test_const_fold_mul(self):
        shape = (6, 6)
        const_tensor1 = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                           vals=np.random.randn(*shape).flatten().astype(np.float32))
        const_tensor2 = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                           vals=np.random.randn(*shape).flatten().astype(np.float32))
        node1 = helper.make_node("Constant", [], ["const1"], value=const_tensor1)
        node2 = helper.make_node("Constant", [], ["const2"], value=const_tensor2)
        node3 = helper.make_node("Mul", ["const1", "const2"], ["mul"])
        node4 = helper.make_node("Mul", ["mul", "X"], ["res"])

        graph = helper.make_graph(
            [node1, node2, node3, node4],
            "test_const_fold_mul",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"X": np.random.randn(*shape).astype(np.float32)}, model_proto,
                             "Mul", 1)

    def test_const_fold_split(self):
        shape = (2, 6, 1)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(2, 6, 1).flatten().astype(np.float32))
        node0 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node1 = helper.make_node("Split", ["const"], ["out1", "out2", "out3"], axis=1)
        node2 = helper.make_node("Sum", ["inp", "out1", "out2", "out3"], ["out4"])

        graph = helper.make_graph(
            [node0, node1, node2],
            "test_const_fold_split",
            [helper.make_tensor_value_info("inp", TensorProto.FLOAT, (2, 2, 1))],
            [helper.make_tensor_value_info("out4", TensorProto.FLOAT, (2, 2, 1))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["out4"], {"inp": np.random.randn(2, 2, 1).astype(np.float32)}, model_proto,
                             "Split", 0)

    def test_const_fold_split_one(self):
        shape = (2, 6, 1)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(2, 6, 1).flatten().astype(np.float32))
        node0 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node1 = helper.make_node("Split", ["const"], ["out1"], axis=1)
        node2 = helper.make_node("Sum", ["inp", "out1"], ["out4"])

        graph = helper.make_graph(
            [node0, node1, node2],
            "test_const_fold_split",
            [helper.make_tensor_value_info("inp", TensorProto.FLOAT, (2, 6, 1))],
            [helper.make_tensor_value_info("out4", TensorProto.FLOAT, (2, 6, 1))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["out4"], {"inp": np.random.randn(2, 6, 1).astype(np.float32)}, model_proto,
                             "Split", 0)

    @check_opset_min_version(13, "Split changed since opset 13")
    def test_const_fold_split_const_splits_13(self):
        shape = (2, 6, 1)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(2, 6, 1).flatten().astype(np.float32))
        node0 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        const_splits = helper.make_tensor(name='const_tensor', data_type=TensorProto.INT64, dims=[3],
                                          vals=np.array([1, 3, 2], np.int64))
        node1 = helper.make_node("Constant", [], ["splits"], value=const_splits)
        node2 = helper.make_node("Split", ["const", "splits"], ["out1", "out2", "out3"], axis=1)
        node3 = helper.make_node("Sum", ["inp", "out2"], ["out4"])

        graph = helper.make_graph(
            [node0, node1, node2, node3],
            "test_const_fold_split",
            [helper.make_tensor_value_info("inp", TensorProto.FLOAT, (2, 3, 1))],
            [helper.make_tensor_value_info("out4", TensorProto.FLOAT, (2, 3, 1))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["out4"], {"inp": np.random.randn(2, 3, 1).astype(np.float32)}, model_proto,
                             "Split", 0)

    @check_opset_max_version(12, "Split changed since opset 13")
    def test_const_fold_split_const_splits(self):
        shape = (2, 6, 1)
        const_tensor = helper.make_tensor(name='const_tensor', data_type=TensorProto.FLOAT, dims=shape,
                                          vals=np.random.randn(2, 6, 1).flatten().astype(np.float32))
        node0 = helper.make_node("Constant", [], ["const"], value=const_tensor)
        node2 = helper.make_node("Split", ["const"], ["out1", "out2", "out3"], axis=1, split=[1, 3, 2])
        node3 = helper.make_node("Sum", ["inp", "out2"], ["out4"])

        graph = helper.make_graph(
            [node0, node2, node3],
            "test_const_fold_split",
            [helper.make_tensor_value_info("inp", TensorProto.FLOAT, (2, 3, 1))],
            [helper.make_tensor_value_info("out4", TensorProto.FLOAT, (2, 3, 1))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["out4"], {"inp": np.random.randn(2, 3, 1).astype(np.float32)}, model_proto,
                             "Split", 0)

    # Const Fold Optimizer Tests End

    # Const Dequantize Optimizer Tests Start

    @check_opset_min_version(10, "DequantizeLinear")
    def test_const_dequantize_reshape(self):
        inputval = numpy_helper.from_array(np.random.randint(0, 100, (2, 3, 4, 5), np.uint8), name='X')
        scale = numpy_helper.from_array(np.array(0.75, dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array(3, dtype=np.uint8), name='zero_point')
        shape = numpy_helper.from_array(np.array([6, 20], dtype=np.int64), name='shape')
        node1 = helper.make_node("DequantizeLinear", ["X", "scale", "zero_point"], ["Y"], name="dequantize")
        node2 = helper.make_node("Reshape", ["Y", "shape"], ["Z"], name="reshape")

        graph = helper.make_graph(
            [node1, node2],
            "const-dequantize-test",
            [],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (6, 20))],
            [inputval, scale, zero_point, shape]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["Z"], {}, model_proto, "Reshape", 0)

    @check_opset_min_version(13, "DequantizeLinear")
    def test_const_dequantize_reshape_per_channel(self):
        inputval = numpy_helper.from_array(np.random.randint(0, 100, (2, 3, 4, 5), np.uint8), name='X')
        scale = numpy_helper.from_array(np.array([0.75, 1., 0.2], dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array([3, 4, 50], dtype=np.uint8), name='zero_point')
        shape = numpy_helper.from_array(np.array([1, 1, 2, 3, 20], dtype=np.int64), name='shape')
        node1 = helper.make_node("DequantizeLinear", ["X", "scale", "zero_point"], ["Y"], name="dequantize", axis=-3)
        node2 = helper.make_node("Reshape", ["Y", "shape"], ["Z"], name="reshape")

        graph = helper.make_graph(
            [node1, node2],
            "const-dequantize-test",
            [],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 1, 2, 3, 20))],
            [inputval, scale, zero_point, shape]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["Z"], {}, model_proto, "Reshape", 0)

    @check_opset_min_version(13, "DequantizeLinear")
    def test_const_dequantize_reshape_per_channel_skipped(self):
        inputval = numpy_helper.from_array(np.random.randint(0, 100, (2, 3, 4, 5), np.uint8), name='X')
        scale = numpy_helper.from_array(np.array([0.75, 1., 0.2, 0.3], dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array([3, 4, 50, 2], dtype=np.uint8), name='zero_point')
        shape = numpy_helper.from_array(np.array([1, 6, 2, 2, 5], dtype=np.int64), name='shape')
        node1 = helper.make_node("DequantizeLinear", ["X", "scale", "zero_point"], ["Y"], name="dequantize", axis=2)
        node2 = helper.make_node("Reshape", ["Y", "shape"], ["Z"], name="reshape")

        graph = helper.make_graph(
            [node1, node2],
            "const-dequantize-test",
            [],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 6, 2, 2, 5))],
            [inputval, scale, zero_point, shape]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        # No optimization can be done here since the channel axis has changed size
        self.run_and_compare(["Z"], {}, model_proto, "Reshape", 1)

    @check_opset_min_version(13, "DequantizeLinear")
    def test_const_dequantize_transpose_per_channel(self):
        inputval = numpy_helper.from_array(np.random.randint(0, 100, (2, 3, 4, 5), np.uint8), name='X')
        scale = numpy_helper.from_array(np.array([0.75, 1., 0.2], dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array([3, 4, 50], dtype=np.uint8), name='zero_point')
        node1 = helper.make_node("DequantizeLinear", ["X", "scale", "zero_point"], ["Y"], name="dequantize", axis=1)
        node2 = helper.make_node("Transpose", ["Y"], ["Z"], name="transpose", perm=[0, 2, 3, 1])

        graph = helper.make_graph(
            [node1, node2],
            "const-dequantize-test",
            [],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 4, 5, 3))],
            [inputval, scale, zero_point]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["Z"], {}, model_proto, "Transpose", 0)

    @check_opset_min_version(13, "DequantizeLinear")
    def test_const_dequantize_unsqueeze_per_channel(self):
        inputval = numpy_helper.from_array(np.random.randint(0, 100, (2, 3, 4, 5), np.uint8), name='X')
        scale = numpy_helper.from_array(np.array([0.75, 1., 0.2], dtype=np.float32), name='scale')
        zero_point = numpy_helper.from_array(np.array([3, 4, 50], dtype=np.uint8), name='zero_point')
        axes = numpy_helper.from_array(np.array([-1, 0, -8, 3, 5], dtype=np.int64), name='axes')
        node1 = helper.make_node("DequantizeLinear", ["X", "scale", "zero_point"], ["Y"], name="dequantize", axis=1)
        node2 = helper.make_node("Unsqueeze", ["Y", "axes"], ["Z"], name="unsqueeze")

        graph = helper.make_graph(
            [node1, node2],
            "const-dequantize-test",
            [],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 1, 2, 1, 3, 1, 4, 5, 1))],
            [inputval, scale, zero_point, axes]
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["Z"], {}, model_proto, "Transpose", 0)

    # Const Dequantize Optimizer Tests End

    def test_transpose_back_to_back_non_const(self):

        node0 = helper.make_node("Transpose", ["u"], ["v"], perm=[0, 2, 3, 1], name="trans_0")
        node1 = helper.make_node("Transpose", ["v"], ["w"], perm=[0, 3, 1, 2], name="trans_1")
        node2 = helper.make_node("Transpose", ["w"], ["x"], perm=[0, 3, 2, 1], name="trans_2")
        node3 = helper.make_node("Transpose", ["x"], ["res"], perm=[1, 3, 0, 2], name="trans_3")

        graph = helper.make_graph(
            [node0, node1, node2, node3],
            "test-transpose-back-to-back-non-const",
            [helper.make_tensor_value_info("u", TensorProto.FLOAT, (5, 5, 5, 5))],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, (5, 5, 5, 5))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {"u": np.random.randn(5, 5, 5, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=1)

    #@check_opset_min_version(9, "string type tensor")
    @unittest.skip("temporarily disabled because of issues with ort-nightly")
    def test_cast_back_to_back_non_const_mixed_types(self):
        node0 = helper.make_node("Cast", ["u"], ["v"], to=11, name="cast_0")  # double
        node1 = helper.make_node("Cast", ["v"], ["w"], to=6, name="cast_1")  # int32
        node2 = helper.make_node("Cast", ["w"], ["x"], to=1, name="cast_2")  # float
        node3 = helper.make_node("Cast", ["x"], ["res"], to=7, name="cast_3")  # int64

        node4 = helper.make_node("Cast", ["w"], ["w2"], to=6, name="cast_4")  # int32
        node5 = helper.make_node("Cast", ["w2"], ["res2"], to=7, name="cast_5")  # int64

        node6 = helper.make_node("Cast", ["x"], ["x2"], to=9, name="cast_6")  # bool
        # TODO: uncomment below after fix
        # https://github.com/microsoft/onnxruntime/issues/2338
        # node7 = helper.make_node("Cast", ["x2"], ["x3"], to=8, name="cast_7")  # string
        node8 = helper.make_node("Cast", ["x2"], ["res3"], to=3, name="cast_8")  # int8

        graph = helper.make_graph(
            [node0, node1, node2, node3, node4, node5, node6, node8],
            "test-cast-back-to-back-non-const",
            [helper.make_tensor_value_info("u", TensorProto.FLOAT, (1, 2, 3))],
            [helper.make_tensor_value_info("res", TensorProto.INT64, (1, 2, 3)),
             helper.make_tensor_value_info("res2", TensorProto.INT64, (1, 2, 3)),
             helper.make_tensor_value_info("res3", TensorProto.INT8, (1, 2, 3))],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res", "res2", "res3"], {"u": np.random.randn(1, 2, 3).astype(np.float32)}, model_proto,
                             "Cast", 5)

    @check_opset_max_version(8, "until opset 8 scales is in attributes")
    def test_upsample_all_ones_removed(self):
        shape = (1, 1, 32, 32)
        node1 = helper.make_node(
            op_type="Upsample",
            inputs=["X"],
            outputs=["Y"],
            scales=[1., 1., 1., 1.],
            name="upsample1")

        graph = helper.make_graph(
            [node1],
            "test_upsample_all_ones",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")

        self.run_and_compare(
            ["Y"],
            {"X": np.random.randn(*shape).astype(np.float32)},
            model_proto,
            "Upsample",
            0)

    @check_opset_min_version(9, ">= 9 scales is in input[1]")
    @check_opset_max_version(9, "Upscale is deprecated in opsets >= 10")
    def test_upsample_all_ones_removed_in_input(self):
        shape = (1, 1, 32, 32)
        const_tensor = helper.make_tensor(
            name="S",
            data_type=TensorProto.FLOAT,
            dims=(1, 4),
            vals=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        node0 = helper.make_node("Constant", [], ["S"], value=const_tensor)
        node1 = helper.make_node(
            op_type="Upsample",
            inputs=["X", "S"],
            outputs=["Y"],
            name="upsample1")

        graph = helper.make_graph(
            [node0, node1],
            "test_upsample_all_ones",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)],
        )

        model_proto = self.make_model(graph, producer_name="onnx-tests")

        self.run_and_compare(
            ["Y"],
            {"X": np.random.randn(*shape).astype(np.float32)},
            model_proto,
            "Upsample",
            0)


if __name__ == "__main__":
    unittest_main()
