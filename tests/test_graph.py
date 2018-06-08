# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import unittest
from collections import namedtuple

import graphviz as gv
import numpy as np
import tensorflow as tf
from onnx import helper

import tf2onnx
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.graph_matcher import *


_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"


def onnx_to_graphviz(g):
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
    model_proto = g.make_model("converted from {}".format(args.input), args.inputs, args.outputs)
    return helper.printable_graph(model_proto.graph)


class Tf2OnnxGraphTests(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        # reset name generation on every test
        tf2onnx.utils.INTERNAL_NAME = 1
        tf.reset_default_graph()
        arg = namedtuple("Arg", "input inputs outputs verbose continue_on_error")
        self._args0 = arg(input="test", inputs=[], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args1 = arg(input="test", inputs=["input:0"], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args2 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args3 = arg(input="test", inputs=["input1:0", "input2:0", "prob:0"], outputs=["output:0"],
                          verbose=False, continue_on_error=False)
        self._args4 = arg(input="test", inputs=["input1:0", "input2:0"], outputs=["output1:0", "output2:0"],
                          verbose=False, continue_on_error=False)

    def test_abs(self):
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 3], name="input")
            x_ = tf.abs(x)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual('digraph { Abs [op_type=Abs] output [op_type=Identity] input:0 -> Abs Abs:0 -> output }',
                             onnx_to_graphviz(g))

    def test_randomuniform(self):
        with tf.Session() as sess:
            shape = tf.constant([2, 3], name="shape")
            x_ = tf.random_uniform(shape, name="rand")
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { RandomUniform__2 [op_type=RandomUniform shape="[2, 3]"] output1 [op_type=Identity] '
                'output2 [op_type=Identity] output [op_type=Identity] RandomUniform__2:0 -> output1 '
                'output1:0 -> output2 output2:0 -> output }',
                onnx_to_graphviz(g))

    def test_randomnormal(self):
        with tf.Session() as sess:
            x_ = tf.random_normal([2, 3], name="rand")
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            actual = onnx_to_graphviz(g)
            expected = 'digraph { RandomNormal__2 [op_type=RandomNormal shape="[2, 3]"] output [op_type=Identity] ' \
                       'RandomNormal__2:0 -> output }'
            self.assertEqual(expected, actual)

    def test_dropout(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x2 = tf.placeholder(tf.float32, [1, 3], name="input2")
            prop = tf.placeholder(tf.float32, name="prob")
            x_ = tf.add(x1, x2)
            x_ = tf.nn.dropout(x_, prop)
            x_ = tf.identity(x_, name="output1")
            x_ = tf.identity(x_, name="output2")
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            actual = onnx_to_graphviz(g)
            expected = 'digraph { Add [op_type=Add] Dropout__3 [op_type=Dropout] output1 [op_type=Identity] ' \
                       'output2 [op_type=Identity] output [op_type=Identity] input1:0 -> Add input2:0 -> ' \
                       'Add Add:0 -> Dropout__3 Dropout__3:0 -> output1 output1:0 -> output2 output2:0 -> output }'
            self.assertEqual(expected, actual)

    def test_add(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x2 = tf.placeholder(tf.float32, [1, 3], name="input2")
            x_ = tf.add(x1, x2)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Add [op_type=Add] output [op_type=Identity] input1:0 -> Add input2:0 -> ' \
                'Add Add:0 -> output }',
                onnx_to_graphviz(g))

    def test_squareddifference(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [1, 3], name="input1")
            x2 = tf.placeholder(tf.float32, [1, 3], name="input2")
            x_ = tf.squared_difference(x1, x2)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { SquaredDifference [op_type=Sub] SquaredDifference__2 [op_type=Mul] '
                'output [op_type=Identity] input1:0 -> SquaredDifference input2:0 -> SquaredDifference '
                'SquaredDifference:0 -> SquaredDifference__2 SquaredDifference:0 -> SquaredDifference__2 '
                'SquaredDifference__2:0 -> output }',
                onnx_to_graphviz(g))

    def test_reducesum(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.reduce_sum(x1)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Sum [op_type=ReduceSum] output [op_type=Identity] input1:0 -> Sum Sum:0 -> output }',
                onnx_to_graphviz(g))

    def test_argminmax(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.argmin(x1, axis=0)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { ArgMin [op_type=ArgMin] output [op_type=Identity] input1:0 -> ArgMin ArgMin:0 -> output }',
                onnx_to_graphviz(g))

    def test_rsqrt(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.rsqrt(x1)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Rsqrt [op_type=Sqrt] Rsqrt__2 [op_type=Reciprocal] output [op_type=Identity] '
                'input1:0 -> Rsqrt Rsqrt:0 -> Rsqrt__2 Rsqrt__2:0 -> output }',
                onnx_to_graphviz(g))

    def test_relu6(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.nn.relu6(x1)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Relu6 [op_type=Max] Relu6__4 [op_type=Min] output [op_type=Identity] input1:0 -> Relu6 '
                'Relu6__2 -> Relu6 Relu6:0 -> Relu6__4 Relu6__3 -> Relu6__4 Relu6__4:0 -> output }',
                onnx_to_graphviz(g))

    def test_conv2d(self):
        kernel = tf.constant([
            [1, 0, 1],
            [2, 1, 0],
            [0, 0, 1]
        ], dtype=tf.float32, name='k')
        kernel = tf.reshape(kernel, [3, 3, 1, 1], name='kernel')

        image = np.array([
            [4, 3, 1, 0],
            [2, 1, 0, 1],
            [1, 2, 4, 1],
            [3, 1, 0, 2]
        ]).reshape([1, 4, 4, 1])

        with tf.Session() as sess:
            image_ = tf.placeholder(tf.float32, shape=image.shape, name='input1')
            conv = tf.nn.conv2d(image_, kernel, strides=[1, 1, 1, 1], padding='VALID')
            _ = tf.identity(conv, name="output")
            sess.run(tf.global_variables_initializer())
            ret = sess.run(conv, feed_dict={image_: image})

            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Conv2D__2 [op_type=Transpose] kernel [op_type=Reshape] Conv2D__3 [op_type=Transpose] ' \
                'Conv2D [op_type=Conv] Conv2D__4 [op_type=Transpose] output [op_type=Identity] input1:0 -> ' \
                'Conv2D__2 k:0 -> kernel "kernel/shape":0 -> kernel kernel:0 -> Conv2D__3 Conv2D__2:0 -> Conv2D ' \
                'Conv2D__3:0 -> Conv2D Conv2D:0 -> Conv2D__4 Conv2D__4:0 -> output }',
                onnx_to_graphviz(g))

    def test_squeeze(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.squeeze(x1)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Squeeze [op_type=Squeeze] output [op_type=Identity] input1:0 -> Squeeze '
                'Squeeze:0 -> output }',
                onnx_to_graphviz(g))

    def test_cast(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.cast(x1, tf.int32)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Cast [op_type=Cast] output [op_type=Identity] input1:0 -> Cast Cast:0 -> output }',
                onnx_to_graphviz(g))

    def test_reshape(self):
        with tf.Session() as sess:
            x1 = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.reshape(x1, [3, 2])
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph)
            self.assertEqual(
                'digraph { Reshape [op_type=Reshape] output [op_type=Identity] input1:0 -> Reshape ' \
                '"Reshape/shape":0 -> Reshape Reshape:0 -> output }',
                onnx_to_graphviz(g))

    def test_custom_rewrite(self):
        # rewriter called from inside process_tf_graph: make a Add a Mul type
        def rewrite_test(g, ops):
            pattern = \
                OpTypePattern('Add', name='op', inputs=["*", "*"])
            ops = g.get_nodes()
            matcher = GraphMatcher(pattern)
            match_results = list(matcher.match_ops(ops))
            for match in match_results:
                op = match.get_op('op')
                op.type = "Mul"
            return ops

        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.add(x, x)
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph, custom_rewriter=[rewrite_test])
            self.assertEqual(
                'digraph { Add [op_type=Mul] output [op_type=Identity] input1:0 -> ' \
                'Add input1:0 -> Add Add:0 -> output }',
                onnx_to_graphviz(g))

    def test_custom_op(self):

        def print_handler(ctx, node, name, args):
            # replace tf.Print() with Identity
            #   T output = Print(T input, data, @list(type) U, @string message, @int first_n, @int summarize)
            # becomes:
            #   T output = Identity(T Input)
            node.type = "Identity"
            node.domain = _TENSORFLOW_DOMAIN
            del node.input[1:]
            return node

        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [2, 3], name="input1")
            x_ = tf.Print(x, [x], "hello")
            _ = tf.identity(x_, name="output")
            g = process_tf_graph(sess.graph,
                                 custom_op_handlers={"Print": print_handler},
                                 extra_opset=helper.make_opsetid(_TENSORFLOW_DOMAIN, 1))
            self.assertEqual(
                'digraph { Print [op_type=Identity] output [op_type=Identity] input1:0 -> Print Print:0 -> output }',
                onnx_to_graphviz(g))


if __name__ == '__main__':
    unittest.main()
