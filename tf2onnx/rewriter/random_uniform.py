# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx random_uniform op
"""
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx import utils


# pylint: disable=missing-docstring


def rewrite_random_uniform(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', inputs=[
                OpTypePattern('RandomUniform', name='input1', inputs=["*"]),
                OpTypePattern('Sub', name='input2', inputs=["*", "*"]),
            ]), None
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        input2 = match.get_op('input2')
        output = match.get_op('output')
        ru_op = match.get_op('input1')
        # max is on input 0
        tmax = input2.inputs[0].get_tensor_value()
        tmin = input2.inputs[1].get_tensor_value()

        new_node = create_onnx_random_uniform_op(g, tmax, tmin, ru_op, output)
        g.replace_all_inputs(ops, output.output[0], new_node.output[0])
        for n in set(match.get_nodes()):
            g.remove_node(n.name)

    return ops


# rewriter function when fold_const is enabled
def rewrite_random_uniform_fold_const(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', name='mul', inputs=[
                OpTypePattern('RandomUniform', name='input1', inputs=["*"]),
                None,
            ]),
            None,
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        output = match.get_op('output')
        mul = match.get_op('mul')
        ru_op = match.get_op('input1')

        tmax_minus_tmin = mul.inputs[1].get_tensor_value()
        tmin = output.inputs[1].get_tensor_value()
        tmax = tmin + tmax_minus_tmin
        new_node = create_onnx_random_uniform_op(g, tmax, tmin, ru_op, output)
        g.replace_all_inputs(ops, output.output[0], new_node.output[0])
        for n in set(match.get_nodes()):
            g.remove_node(n.name)

    return ops


def create_onnx_random_uniform_op(g, tmax, tmin, ru_op, output):
    dtype = g.get_dtype(output.output[0])
    op_name = utils.make_name("RandomUniform")
    if ru_op.inputs[0].type == "Shape":
        shape_node = ru_op.inputs[0]
        new_node = g.make_node("RandomUniformLike", inputs=[shape_node.input[0]], name=op_name,
                               attr={"low": tmin, "high": tmax, "dtype": dtype},
                               shapes=shape_node.output_shapes, dtypes=[dtype])
    else:
        shape = g.get_shape(output.output[0])
        new_node = g.make_node("RandomUniform", [], name=op_name,
                               attr={"low": tmin, "high": tmax, "dtype": dtype, "shape": shape},
                               shapes=[shape], dtypes=[dtype])
    return new_node
