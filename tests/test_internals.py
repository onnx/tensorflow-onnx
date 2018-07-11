# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import division
from __future__ import print_function

import unittest
from collections import namedtuple

import graphviz as gv
from onnx import TensorProto
from onnx import helper

import tf2onnx
import tf2onnx.utils
from tf2onnx.graph import Node, Graph
from tf2onnx.graph_matcher import *


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
    model_proto = g.make_model("converted from {}".format(args.input), args.inputs, args.outputs)
    return helper.printable_graph(model_proto.graph)


class Tf2OnnxInternalTests(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        # reset name generation on every test
        tf2onnx.utils.INTERNAL_NAME = 1
        arg = namedtuple("Arg", "input inputs outputs verbose")
        self._args0 = arg(input="test", inputs=[], outputs=["output:0"], verbose=False)
        self._args1 = arg(input="test", inputs=["input:0"], outputs=["output:0"], verbose=False)
        self._args2 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output:0"], verbose=False)
        self._args3 = arg(input="test", inputs=["input1:0", "input2:0", "prob:0"], outputs=["output:0"], verbose=False)
        self._args4 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output1:0", "output2:0"],
                          verbose=False)

    @staticmethod
    def sample_net():
        n1 = helper.make_node("Abs", ["input"], ["n1:0"], name="n1")
        n2 = helper.make_node("Abs", ["n1:0"], ["n2:0"], name="n2")
        n3 = helper.make_node("Abs", ["n1:0"], ["n3:0"], name="n3")
        n4 = helper.make_node("Add", ["n2:0", "n3:0"], ["n4:0"], name="n4")
        n5 = helper.make_node("Abs", ["n4:0"], ["n5:0"], name="n5")
        n6 = helper.make_node("Identity", ["n5:0"], ["n6:0"], name="n6")

        model_proto = helper.make_graph(
            nodes=[n1, n2, n3, n4, n5, n6],
            name="test",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("n5:0", TensorProto.FLOAT, [2, 2])],
            initializer=[]
        )
        return model_proto

    def test_insert_node1(self):
        model_proto = self.sample_net()
        nodes = model_proto.node
        g = Graph(nodes, output_shapes={}, dtypes={})
        n2 = g.get_node_by_name("n2")
        n7 = g.insert_new_node_on_input(n2, "Abs", "n1:0", name="n7")
        ops = g.get_nodes()
        ops.append(n7)
        g.topological_sort(ops)
        result = onnx_to_graphviz(g)
        expected = 'digraph { n1 [op_type=Abs] n7 [op_type=Abs] n2 [op_type=Abs] n3 [op_type=Abs] ' \
                   'n4 [op_type=Add] n5 [op_type=Abs] n6 [op_type=Identity] ' \
                   'input -> n1 n1:0 -> n7 n7:0 -> n2 n1:0 -> n3 n2:0 -> n4 n3:0 -> n4 n4:0 -> n5 n5:0 -> n6 }'
        self.assertEqual(expected, result)

    def test_insert_node2(self):
        model_proto = self.sample_net()
        nodes = model_proto.node
        g = Graph(nodes, output_shapes={}, dtypes={})
        n7 = g.insert_new_node_on_output("Abs", "n1:0", name="n7")
        ops = g.get_nodes()
        ops.append(n7)
        g.topological_sort(ops)
        result = onnx_to_graphviz(g)
        expected = 'digraph { n1 [op_type=Abs] n7 [op_type=Abs] n3 [op_type=Abs] n2 [op_type=Abs] ' \
                   'n4 [op_type=Add] n5 [op_type=Abs] n6 [op_type=Identity] ' \
                   'input -> n1 n1:0 -> n7 n7:0 -> n3 n7:0 -> n2 n2:0 -> n4 n3:0 -> n4 n4:0 -> n5 n5:0 -> n6 }'
        self.assertEqual(expected, result)

    def test_remove_input(self):
        model_proto = self.sample_net()
        nodes = model_proto.node
        g = Graph(nodes, output_shapes={}, dtypes={})
        n4 = g.get_node_by_name("n4")
        g.remove_input(n4, n4.input[1])
        result = onnx_to_graphviz(g)
        expected = 'digraph { n1 [op_type=Abs] n2 [op_type=Abs] n3 [op_type=Abs] n4 [op_type=Add] ' \
                   'n5 [op_type=Abs] n6 [op_type=Identity] input -> n1 n1:0 -> n2 n1:0 -> n3 n2:0 -> n4 ' \
                   'n4:0 -> n5 n5:0 -> n6 }'
        self.assertEqual(expected, result)

    def test_rewrite_subgraph(self):
        model_proto = self.sample_net()
        nodes = model_proto.node
        g = tf2onnx.graph.Graph(nodes, output_shapes={}, dtypes={})
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
            op_name = tf2onnx.utils.make_name("ReplacedOp")
            out_name = op_name + ":0"
            new_node = Node(helper.make_node("Sub", input_node.input, [out_name], name=op_name), g)
            ops = g.replace_subgraph(ops, match, [], [output_node], [], [new_node])
        g.topological_sort(ops)
        result = onnx_to_graphviz(g)
        expected = 'digraph { n1 [op_type=Abs] n3 [op_type=Abs] n2 [op_type=Abs] ReplacedOp__2 [op_type=Sub] ' \
                   'n6 [op_type=Identity] input -> n1 n1:0 -> n3 n1:0 -> n2 n2:0 -> ReplacedOp__2 ' \
                   'n3:0 -> ReplacedOp__2 ReplacedOp__2:0 -> n6 }'
        self.assertEqual(expected, result)

    def test_match_flipped(self):
        n1 = helper.make_node("Sub", ["i1", "i1"], ["n1:0"], name="n1")
        n2 = helper.make_node("Add", ["i2", "i2"], ["n2:0"], name="n2")
        n3 = helper.make_node("Mul", ["n1:0", "n2:0"], ["n3:0"], name="n3")

        model_proto = helper.make_graph(
            nodes=[n1, n2, n3],
            name="test",
            inputs=[helper.make_tensor_value_info("i1", TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info("i2", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("n2:0", TensorProto.FLOAT, [2, 2])],
            initializer=[]
        )
        nodes = model_proto.node
        g = tf2onnx.graph.Graph(nodes, output_shapes={}, dtypes={})
        pattern = OpTypePattern('Mul', inputs=[
            OpTypePattern('Add'),
            OpTypePattern('Sub')
        ])
        ops = g.get_nodes()
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        self.assertEqual(1, len(match_results))


if __name__ == '__main__':
    unittest.main()
