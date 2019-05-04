# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for internal methods."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import graphviz as gv
import numpy as np
from onnx import TensorProto
from onnx import helper, numpy_helper

import tensorflow as tf
from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.graph import GraphUtil

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring

def onnx_to_graphviz(g):
    """Onnx graph as dot string."""
    g2 = gv.Digraph()
    for node in g.get_nodes():
        kwarg = {}
        attr = node.attr
        if "shape" in attr:
            kwarg["shape"] = str(attr["shape"].ints)
        if "broadcast" in attr:
            kwarg["broadcast"] = str(attr["broadcast"].i)
        g2.node(node.name, op_type=node.type, **kwarg)
    for node in g.get_nodes():
        for i in node.input:
            if i:
                g2.edge(i, node.name)
    return " ".join(g2.source.split())


def onnx_pretty(g, args=None):
    """Onnx graph pretty print."""
    graph_proto = g.make_model("converted from {}".format(args.input))
    return helper.printable_graph(graph_proto.graph)


class Tf2OnnxInternalTests(Tf2OnnxBackendTestBase):
    def setUp(self):
        super().setUp()
        arg = namedtuple("Arg", "input inputs outputs")
        self._args0 = arg(input="test", inputs=[], outputs=["output:0"])
        self._args1 = arg(input="test", inputs=["input:0"], outputs=["output:0"])
        self._args2 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output:0"])
        self._args3 = arg(input="test", inputs=["input1:0", "input2:0", "prob:0"], outputs=["output:0"])
        self._args4 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output1:0", "output2:0"])

    @staticmethod
    def sample_net():
        n1 = helper.make_node("Abs", ["input"], ["n1:0"], name="n1")
        n2 = helper.make_node("Abs", ["n1:0"], ["n2:0"], name="n2")
        n3 = helper.make_node("Abs", ["n1:0"], ["n3:0"], name="n3")
        n4 = helper.make_node("Add", ["n2:0", "n3:0"], ["n4:0"], name="n4")
        n5 = helper.make_node("Abs", ["n4:0"], ["n5:0"], name="n5")
        n6 = helper.make_node("Identity", ["n5:0"], ["n6:0"], name="n6")

        graph_proto = helper.make_graph(
            nodes=[n1, n2, n3, n4, n5, n6],
            name="test",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("n5:0", TensorProto.FLOAT, [2, 2])],
            initializer=[]
        )
        return graph_proto

    def test_insert_node1(self):
        graph_proto = self.sample_net()
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        n2 = g.get_node_by_name("n2")
        g.insert_new_node_on_input(n2, "Abs", "n1:0", name="n7")
        ops = g.get_nodes()
        g.topological_sort(ops)
        result = onnx_to_graphviz(g)
        expected = 'digraph { Placeholder__4 [op_type=Placeholder] ' \
                   'n1 [op_type=Abs] n7 [op_type=Abs] n2 [op_type=Abs] n3 [op_type=Abs] ' \
                   'n4 [op_type=Add] n5 [op_type=Abs] n6 [op_type=Identity] ' \
                   'n5_graph_outputs_Identity__3 [op_type=Identity] input -> n1 n1:0 -> n7 ' \
                   'n7:0 -> n2 n1:0 -> n3 n2:0 -> n4 n3:0 -> n4 n4:0 -> n5 n5_raw_output___2:0 -> n6 ' \
                   'n5_raw_output___2:0 -> n5_graph_outputs_Identity__3 }'
        self.assertEqual(expected, result)

    def test_insert_node2(self):
        graph_proto = self.sample_net()
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        g.insert_new_node_on_output("Abs", "n1:0", name="n7")
        ops = g.get_nodes()
        g.topological_sort(ops)
        result = onnx_to_graphviz(g)
        expected = 'digraph { Placeholder__4 [op_type=Placeholder] n1 [op_type=Abs] n7 [op_type=Abs] ' \
                   'n3 [op_type=Abs] n2 [op_type=Abs] n4 [op_type=Add] n5 [op_type=Abs] ' \
                   'n6 [op_type=Identity] n5_graph_outputs_Identity__3 [op_type=Identity] ' \
                   'input -> n1 n1:0 -> n7 n7:0 -> n3 n7:0 -> n2 n2:0 -> n4 n3:0 -> n4 n4:0 -> n5 ' \
                   'n5_raw_output___2:0 -> n6 n5_raw_output___2:0 -> n5_graph_outputs_Identity__3 }'
        self.assertEqual(expected, result)

    def test_remove_input(self):
        graph_proto = self.sample_net()
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        n4 = g.get_node_by_name("n4")
        g.remove_input(n4, n4.input[1])
        ops = g.get_nodes()
        g.topological_sort(ops)
        result = onnx_to_graphviz(g)
        expected = 'digraph { Placeholder__4 [op_type=Placeholder] n1 [op_type=Abs] n3 [op_type=Abs] ' \
                   'n2 [op_type=Abs] n4 [op_type=Add] n5 [op_type=Abs] n6 [op_type=Identity] ' \
                   'n5_graph_outputs_Identity__3 [op_type=Identity] input -> n1 n1:0 -> n3 ' \
                   'n1:0 -> n2 n2:0 -> n4 n4:0 -> n5 n5_raw_output___2:0 -> n6 ' \
                   'n5_raw_output___2:0 -> n5_graph_outputs_Identity__3 }'
        self.assertEqual(expected, result)

    def test_rewrite_subgraph(self):
        graph_proto = self.sample_net()
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        pattern = \
            OpTypePattern('Abs', name='output', inputs=[
                OpTypePattern('Add', name='input')
            ])
        ops = g.get_nodes()
        matcher = GraphMatcher(pattern)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            input_node = match.get_op('input')
            output_node = match.get_op('output')
            op_name = utils.make_name("ReplacedOp")
            out_name = utils.port_name(op_name)
            new_node = g.make_node("Sub", inputs=input_node.input, outputs=[out_name], name=op_name)
            g.replace_all_inputs(ops, output_node.output[0], new_node.output[0])
            for n in set(match.get_nodes()):
                g.remove_node(n.name)
        g.topological_sort(ops)
        result = onnx_to_graphviz(g)
        expected = 'digraph { Placeholder__4 [op_type=Placeholder] n1 [op_type=Abs] ' \
                   'n3 [op_type=Abs] n2 [op_type=Abs] ReplacedOp__5 [op_type=Sub] ' \
                   'n6 [op_type=Identity] n5_graph_outputs_Identity__3 [op_type=Identity] ' \
                   'input -> n1 n1:0 -> n3 n1:0 -> n2 n2:0 -> ReplacedOp__5 n3:0 -> ReplacedOp__5 ' \
                   'ReplacedOp__5:0 -> n6 ReplacedOp__5:0 -> n5_graph_outputs_Identity__3 }'
        self.assertEqual(expected, result)

    def test_match_flipped(self):
        n1 = helper.make_node("Sub", ["i1", "i1"], ["n1:0"], name="n1")
        n2 = helper.make_node("Add", ["i2", "i2"], ["n2:0"], name="n2")
        n3 = helper.make_node("Mul", ["n1:0", "n2:0"], ["n3:0"], name="n3")

        graph_proto = helper.make_graph(
            nodes=[n1, n2, n3],
            name="test",
            inputs=[helper.make_tensor_value_info("i1", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("i2", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("n2:0", TensorProto.FLOAT, [2, 2])],
            initializer=[]
        )
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        pattern = OpTypePattern('Mul', inputs=[
            OpTypePattern('Add'),
            OpTypePattern('Sub')
        ])
        ops = g.get_nodes()
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        self.assertEqual(1, len(match_results))

    def test_cmdarg_parse(self):
        arg = "input/V-1_2:0,input/X:0[1,2,3],Y:1[4,5],Z:3,A:1,B"
        expected_inputs = ['input/V-1_2:0', 'input/X:0', 'Y:1', 'Z:3', 'A:1', 'B']
        expected_shape = {'Y:1': [4, 5], 'input/X:0': [1, 2, 3]}
        inputs, shape_override = utils.split_nodename_and_shape(arg)
        self.assertEqual(expected_inputs, inputs)
        self.assertEqual(expected_shape, shape_override)

    def test_shape_utils(self):
        self.assertEqual(utils.merge_shapes(None, None), None)
        self.assertEqual(utils.merge_shapes([], None), [])
        self.assertEqual(utils.merge_shapes(None, [1, 2, 3]), [1, 2, 3])
        self.assertEqual(utils.merge_shapes([1, 3], [None, 3]), [1, 3])
        self.assertEqual(utils.merge_shapes([1, None, 3], (-1, 2, "unk")), [1, 2, 3])

        self.assertTrue(utils.are_shapes_compatible(None, []))
        self.assertTrue(utils.are_shapes_compatible([1, None, 3], (-1, 2, "unk")))
        self.assertFalse(utils.are_shapes_compatible([1, 2, 3], (2, 3)))
        self.assertFalse(utils.are_shapes_compatible([1, 2, 3], (4, 5, 6)))

        self.assertTrue(utils.are_shapes_equal(None, None))
        self.assertFalse(utils.are_shapes_equal(None, []))
        self.assertTrue(utils.are_shapes_equal([1, 2, 3], (1, 2, 3)))

    def test_data_format(self):
        n1 = helper.make_node("Conv", ["X", "W"], ["Y"], name="n1", data_format="NHWC")
        graph_proto = helper.make_graph(
            nodes=[n1],
            name="test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
            initializer=[]
        )
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        n = g.get_node_by_name("n1")
        self.assertEqual(n.data_format, "NHWC")
        self.assertTrue(n.is_nhwc())

    def test_node_attr_onnx(self):
        n1 = helper.make_node("Conv", ["X", "W"], ["Y"], name="n1", my_attr="my_attr")
        graph_proto = helper.make_graph(
            nodes=[n1],
            name="test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
            initializer=[]
        )
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        n1 = g.get_node_by_name("n1")
        self.assertTrue("my_attr" in n1.attr)
        self.assertTrue("my_attr" not in n1.attr_onnx)

        n1 = helper.make_node("Conv", ["X", "W"], ["Y"], name="n1", domain="my_domain", my_attr="my_attr")
        graph_proto = helper.make_graph(
            nodes=[n1],
            name="test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
            initializer=[]
        )
        g = GraphUtil.create_graph_from_onnx_graph(graph_proto)
        n1 = g.get_node_by_name("n1")
        self.assertTrue("my_attr" in n1.attr)
        self.assertTrue("my_attr" in n1.attr_onnx)

    def test_tensor_data(self):
        tensors = {
            "empty_tensor": np.array([], dtype=np.float32),
            "multi_dim_empty_tensor": np.array([[], []], dtype=np.float32),
            "scalar": np.array(1., dtype=np.float32),
            "one_item_array": np.array([1.], dtype=np.float32),
            "normal_array": np.array([[1., 2.], [2., 3.]], dtype=np.float32)
        }
        tf.reset_default_graph()
        with tf.Session() as sess:
            for n, data in tensors.items():
                tf.constant(data, dtype=tf.float32, name=n)

        for tf_node in sess.graph.get_operations():
            name = tf_node.name
            self.assertTrue(name in tensors.keys())

            self.assertTrue("value" in tf_node.node_def.attr)
            # convert to onnx tensor value
            tensor_value = utils.tf_to_onnx_tensor(
                utils.get_tf_node_attr(tf_node, "value"),
                name=utils.port_name(tf_node.name)
            )
            attr = helper.make_attribute("value", tensor_value)
            # same as node.get_tensor_value(is_list=False)
            actual = numpy_helper.to_array(helper.get_attribute_value(attr))

            expected = tensors[name]

            self.assertTrue(np.array_equal(expected, actual))


if __name__ == '__main__':
    unittest_main()
