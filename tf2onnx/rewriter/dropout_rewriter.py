# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx dropout op
"""

from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_dropout(g, ops):
    patterns = [
        OpTypePattern('Mul', name='outputs', inputs=[
            OpTypePattern('RealDiv', name="input2"),
            OpTypePattern('Floor', inputs=[
                OpTypePattern('Add', inputs=[
                    OpTypePattern(None, name="input3"),
                    OpTypePattern('RandomUniform|RandomUniformLike'),
                ])
            ]),
        ]),
        OpTypePattern("Mul", name="outputs", inputs=[
            OpTypePattern("Mul", name="input2"),
            OpTypePattern("Cast", inputs=[
                OpTypePattern("GreaterEqual", inputs=[
                    OpTypePattern("RandomUniform|RandomUniformLike"),
                    OpTypePattern(None, name="input3")
                ])
            ])
        ]),
        # pattern for tf-2.0 tf.nn.dropout()
        OpTypePattern("Mul", name="outputs", inputs=[
            OpTypePattern("Cast", inputs=[
                OpTypePattern("GreaterEqual", inputs=[
                    OpTypePattern("RandomUniform|RandomUniformLike"),
                    OpTypePattern(None, name="input3")
                ])
            ]),
            OpTypePattern("Mul", name="input2"),
        ]),
    ]
    for pattern in patterns:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            inputs2 = match.get_op('input2')
            if inputs2.inputs[0].type == "RealDiv":
                data = inputs2.input[1]
            else:
                data = inputs2.input[0]
            outputs = match.get_op('outputs')
            op_name = utils.make_name("Dropout")
            out_name = utils.port_name(op_name)
            new_node = g.make_node(
                "Dropout",
                [data],
                outputs=[out_name],
                name=op_name,
                attr={"ratio": 1.0},
                shapes=[g.get_shape(inputs2.input[0])],
                dtypes=[g.get_dtype(inputs2.input[0])]
            )

            g.replace_all_inputs(ops, outputs.output[0], new_node.output[0])
            g.safe_remove_nodes(match.get_nodes())
            if outputs.output[0] in g.outputs:
                g.make_node('Identity', [new_node.output[0]], outputs=[outputs.output[0]])

    # remove dropout if its ratio is 1.0
    for node in g.get_nodes():
        if node.type == "Dropout" and node.get_attr("ratio").f == 1.0:
            g.replace_all_inputs(g.get_nodes(), node.output[0], node.input[0])
            g.remove_node(node.name)

    return ops
