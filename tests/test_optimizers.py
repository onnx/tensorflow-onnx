# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for optimizers such as TransposeOptimizer."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from onnx import helper, TensorProto
from tf2onnx import utils
from tf2onnx.graph import GraphUtil
from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main, group_nodes_by_type


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class OptimizerTests(Tf2OnnxBackendTestBase):
    """Run original model proto and modified model proto with onnxruntime, compare the results."""

    def run_and_compare(self, output_names_with_port, onnx_feed_dict, origin_proto, op_type,
                        remaining_op_num, debug=False, rtol=1e-07):
        utils.make_sure(op_type is not None, "op_type should be specified")
        utils.make_sure(remaining_op_num is not None, "remaining_op_num should be specified")

        origin_model_path = self.save_onnx_model(origin_proto, onnx_feed_dict, postfix="_origin")

        new_proto = GraphUtil.optimize_model_proto(origin_proto)

        self.assertTrue(new_proto, msg="model proto after optimizer should not be None")

        new_model_path = self.save_onnx_model(new_proto, onnx_feed_dict, postfix="_opt")
        current = GraphUtil.get_node_count_from_onnx_graph(new_proto.graph)

        self.assertTrue(current[op_type] == remaining_op_num,
                        msg="Expect " + str(remaining_op_num) + " " + op_type + " ops left, but actually " + str(
                            current[op_type]) + " left")

        if self.config.is_onnxruntime_backend:
            expected = self.run_onnxruntime(origin_model_path, onnx_feed_dict, output_names_with_port)
            actual = self.run_onnxruntime(new_model_path, onnx_feed_dict, output_names_with_port)
        else:
            raise ValueError("only onnxruntime is supported to test transpose optimizer")

        for expected_val, actual_val in zip(expected, actual):
            self.assertAllClose(expected_val, actual_val, rtol=rtol, atol=1e-5)
            self.assertEqual(expected_val.dtype, actual_val.dtype)
            self.assertEqual(expected_val.shape, actual_val.shape)

        return new_proto
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

    def test_transpose_with_concat(self):
        input_shape = (2, 3, 4, 5)
        perm = [0, 3, 1, 2]
        input_shape_with_trans = [input_shape[i] for i in perm]
        for axis in [0, 1, 2, 3]:
            output_before_trans = list(input_shape)
            output_before_trans[axis] *= 2
            output_shape = [output_before_trans[i] for i in [0, 3, 1, 2]]
            node1 = helper.make_node("Transpose", ["input_data1"], ["Y"], perm=[0, 2, 3, 1], name="trans")
            node2 = helper.make_node("Concat", ["Y", "input_data2"], ["Z"], axis=axis, name="concat")
            node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans2")

            graph = helper.make_graph(
                [node1, node2, node3],
                "test_transpose_with_concat",
                [helper.make_tensor_value_info("input_data1", TensorProto.FLOAT, input_shape_with_trans),
                 helper.make_tensor_value_info("input_data2", TensorProto.FLOAT, input_shape),
                 ],
                [helper.make_tensor_value_info("res", TensorProto.FLOAT, output_shape)],
            )

            model_proto = helper.make_model(graph, producer_name="onnx-tests")
            feed_dict = {"input_data1": np.random.randn(*input_shape_with_trans).astype(np.float32),
                         "input_data2": np.random.randn(*input_shape).astype(np.float32),
                         }
            self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=1)

    def test_transpose_with_add1(self):
        # when transpose follows with a broadcasting op
        # reshape is needed when switching transpose with this op and op need broadcast its inputs
        node1 = helper.make_node("Transpose", ["input_data1"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Add", ["Y", "input_data2"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "transpose_with_shape",
            [helper.make_tensor_value_info("input_data1", TensorProto.FLOAT, (2, 3, 4, 5)),
             helper.make_tensor_value_info("input_data2", TensorProto.FLOAT, (3,)),
             ],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        feed_dict = {"input_data1": np.random.randn(2, 3, 4, 5).astype(np.float32),
                     "input_data2": np.random.randn(3).astype(np.float32),
                     }
        self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=0)

    def test_transpose_with_add2(self):
        node1 = helper.make_node("Transpose", ["input_data1"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Add", ["Y", "input_data2"], ["Z"], name="add")
        node3 = helper.make_node("Transpose", ["Z"], ["res"], perm=[0, 3, 1, 2], name="trans2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "transpose_with_shape",
            [helper.make_tensor_value_info("input_data1", TensorProto.FLOAT, (2, 3, 4, 5)),
             helper.make_tensor_value_info("input_data2", TensorProto.FLOAT, (2, 4, 5, 3)),
             ],
            [helper.make_tensor_value_info("res", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        feed_dict = {"input_data1": np.random.randn(2, 3, 4, 5).astype(np.float32),
                     "input_data2": np.random.randn(2, 4, 5, 3).astype(np.float32),
                     }
        self.run_transpose_compare(["res"], feed_dict, model_proto, remaining_transpose_num=1)

    def test_transpose_relu(self):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node2 = helper.make_node("Relu", ["Y"], ["Z"], name="relu")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "relu-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_transpose_leaky_relu(self):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node2 = helper.make_node("LeakyRelu", ["Y"], ["Z"], alpha=0.02, name="relu")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [node1, node2, node3],
            "LeakyRelu-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_transpose_max(self):
        const_1_val = [2.0]
        const_1 = helper.make_tensor("const_1", TensorProto.FLOAT, (1,), const_1_val)
        const_1_node = helper.make_node("Constant", [], ["const_1"], value=const_1, name="const_1")

        const_2_val = np.random.randn(2, 4, 5, 3).astype(np.float32).reshape(120).tolist()
        const_2 = helper.make_tensor("const_2", TensorProto.FLOAT, (2, 4, 5, 3), const_2_val)
        const_2_node = helper.make_node("Constant", [], ["const_2"], value=const_2, name="const_2")

        const_3_val = np.random.randn(2, 4, 5, 3).astype(np.float32).reshape(120).tolist()
        const_3 = helper.make_tensor("const_3", TensorProto.FLOAT, (2, 4, 5, 3), const_3_val)
        const_3_node = helper.make_node("Constant", [], ["const_3"], value=const_3, name="const_3")

        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans_1")
        node2 = helper.make_node("Max", ["Y", "const_3", "const_2", "const_1"], ["Z"], name="max")
        node3 = helper.make_node("Transpose", ["Z"], ["Z1"], perm=[0, 3, 1, 2], name="trans_2")

        graph = helper.make_graph(
            [const_1_node, const_2_node, const_3_node, node1, node2, node3],
            "Max-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (2, 3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_transpose_merge(self):
        node0 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node1 = helper.make_node("Transpose", ["X"], ["Y_1"], perm=[0, 2, 3, 1], name="trans_1")
        node2 = helper.make_node("Mul", ["Y", "Y_1"], ["OUT"], name="mul")

        graph = helper.make_graph(
            [node0, node1, node2],
            "transpose-merge-test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, (2, 4, 5, 3))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["OUT"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=1)

    def test_transpose_with_shape(self):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Shape", ["Y"], ["Z"], name="shape")

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_shape",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.INT64, [4])],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_transpose_with_identity(self):
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Identity", ["Y"], ["Z"], name="identity")

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_identity",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 4, 5, 3))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=1)

    def test_transpose_with_squeeze1(self):
        # squeeze the first dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[0])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 3, 4, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (4, 5, 3))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        model_after_opt = self.run_transpose_compare(["Z"], {"X": np.random.randn(1, 3, 4, 5).astype(np.float32)},
                                                     model_proto, remaining_transpose_num=1)
        self.check_transpose_perm(model_after_opt, [1, 2, 0])

    def test_transpose_with_squeeze2(self):
        # squeeze the second dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[1])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (3, 4, 1, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 5, 4))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        model_after_opt = self.run_transpose_compare(["Z"], {"X": np.random.randn(3, 4, 1, 5).astype(np.float32)},
                                                     model_proto, remaining_transpose_num=1)
        self.check_transpose_perm(model_after_opt, [0, 2, 1])

    def test_transpose_with_squeeze3(self):
        # squeeze the last dim
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[3])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (3, 1, 4, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 4, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(3, 1, 4, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_transpose_with_squeeze4(self):
        # squeeze the two dims
        node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 3, 1], name="trans")
        node2 = helper.make_node("Squeeze", ["Y"], ["Z"], name="squeeze", axes=[1, 3])

        graph = helper.make_graph(
            [node1, node2],
            "transpose_with_squeeze",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (3, 1, 1, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 5))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["Z"], {"X": np.random.randn(3, 1, 1, 5).astype(np.float32)},
                                   model_proto, remaining_transpose_num=0)

    def test_trans_output_as_graph_outputs(self):
        """
        If transpose's output is graph's output, don't optimize it.
        """
        trans = helper.make_node("Transpose", ["X"], ["Y"], name="trans", perm=[0, 2, 3, 1])
        graph_proto = helper.make_graph(
            [trans],
            "trans-to-graph-output",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4, 5))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4, 5, 3))],
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

    def test_trans_can_be_replaced_with_reshape1(self):
        # test trans-NHWC
        input_shapes_np = [(2, 3, 4, 1), (2, 1, 1, 4), (2, 3, 4, 1)]
        input_shapes = [(2, 3, 4, 1), (2, 1, 1, 4), (2, -1, -1, 1)]
        perm = (0, 3, 1, 2)
        for input_shape_np, input_shape in zip(input_shapes_np, input_shapes):
            result_shape = [input_shape[i] for i in perm]
            node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
            graph = helper.make_graph(
                [node1],
                "test_trans_can_be_replaced_with_reshape",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, result_shape)],
            )

            model_proto = helper.make_model(graph, producer_name="onnx-tests")
            self.run_transpose_compare(["Y"], {"X": np.random.randn(*input_shape_np).astype(np.float32)},
                                       model_proto, remaining_transpose_num=0)

    def test_trans_can_be_replaced_with_reshape2(self):
        # test trans-NCHW
        input_shapes_np = [(2, 1, 3, 4), (2, 4, 1, 1), (2, 1, 3, 4)]
        input_shapes = [(2, 1, 3, 4), (2, 4, 1, 1), (2, 1, -1, -1)]
        perm = (0, 2, 3, 1)
        for input_shape_np, input_shape in zip(input_shapes_np, input_shapes):
            result_shape = [input_shape[i] for i in perm]
            node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=perm, name="trans")
            graph = helper.make_graph(
                [node1],
                "test_trans_can_be_replaced_with_reshape",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, result_shape)],
            )

            model_proto = helper.make_model(graph, producer_name="onnx-tests")
            self.run_transpose_compare(["Y"], {"X": np.random.randn(*input_shape_np).astype(np.float32)},
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
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
                vals=iter_num_value.flatten().astype(np.int64),
            ),
        )

        cond_value = np.array(True, dtype=np.bool)
        node3 = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['cond_value'],
            value=helper.make_tensor(
                name='cond_value',
                data_type=TensorProto.BOOL,
                dims=iter_num_value.shape,
                vals=cond_value.flatten().astype(np.bool),
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_identity_compare(["Z1"], {"X": np.random.randn(2, 3, 4, 5).astype(np.float32)},
                                  model_proto, remaining_identity_num=0)

    # Identity Optimizer Tests End

    # Merge Duplicated Nodes Optimizer Tests Start

    def run_merge_duplicated_nodes_compare(self, output_names_with_port, onnx_feed_dict, origin_proto,
                                           op_type=None, remaining_op_num=None, debug=False, rtol=1e-07):
        self.run_and_compare(output_names_with_port, onnx_feed_dict, origin_proto, op_type=op_type,
                             remaining_op_num=remaining_op_num, debug=debug, rtol=rtol)

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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["OUT"], {"X": np.random.randn(5, 5).astype(np.float32)}, model_proto,
                                                op_type="Add", remaining_op_num=2)

    def test_duplicated_duplicated_attributes(self):
        # same attr or not
        node0 = helper.make_node('ReduceSum', inputs=["X"], outputs=["value0"], axes=[0], keepdims=0)
        node1 = helper.make_node('ReduceSum', inputs=["X"], outputs=["value1"], axes=[0], keepdims=0)
        node2 = helper.make_node('ReduceSum', inputs=["X"], outputs=["value2"], axes=[1], keepdims=0)
        node3 = helper.make_node('Add', inputs=["value0", "value1"], outputs=["value3"])
        node4 = helper.make_node("Mul", ["value2", "value3"], ["OUT"])

        graph = helper.make_graph(
            [node0, node1, node2, node3, node4],
            "test_duplicated_duplicated_attributes",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5))],
            [helper.make_tensor_value_info("OUT", TensorProto.FLOAT, (5,))],
        )

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["OUT"], {"X": np.random.randn(5, 5).astype(np.float32)}, model_proto,
                                                op_type="ReduceSum", remaining_op_num=2)

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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["value1", "value2"],
                                                {"X": np.random.randn(5, 5).astype(np.float32)}, model_proto,
                                                op_type="Add", remaining_op_num=2)

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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_merge_duplicated_nodes_compare(["res"], {"X": np.random.randn(5).astype(np.float32)},
                                                model_proto,
                                                op_type="Log", remaining_op_num=3)

    # Merge Duplicated Nodes Optimizer Tests End

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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_transpose_compare(["res"], {},
                                   model_proto, remaining_transpose_num=0)

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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
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

        model_proto = helper.make_model(graph, producer_name="onnx-tests")
        self.run_and_compare(["res"], {"X": np.random.randn(*shape).astype(np.int64)}, model_proto,
                             "Cast", 0)
    # Const Fold Optimizer Tests End


if __name__ == "__main__":
    unittest_main()
