# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx dropout op
"""

from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx import logging

logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring


def rewrite_dropout(g, ops):
    patterns = [
        OpTypePattern('Mul', name='outputs', inputs=[
            OpTypePattern('RealDiv', name="input2"),
            OpTypePattern('Floor', inputs=[
                OpTypePattern('Add', inputs=[
                    OpTypePattern("*", name="input3"),
                    OpTypePattern('RandomUniform|RandomUniformLike'),
                ])
            ]),
        ]),
        OpTypePattern("Mul", name="outputs", inputs=[
            OpTypePattern("Mul", name="input2"),
            OpTypePattern("Cast", inputs=[
                OpTypePattern("GreaterEqual", inputs=[
                    OpTypePattern("RandomUniform|RandomUniformLike"),
                    OpTypePattern("*", name="input3")
                ])
            ])
        ]),
        # pattern for tf-2.0 tf.nn.dropout()
        OpTypePattern("Mul", name="outputs", inputs=[
            OpTypePattern("Cast", inputs=[
                OpTypePattern("GreaterEqual", inputs=[
                    OpTypePattern("RandomUniform|RandomUniformLike"),
                    OpTypePattern("*", name="input3")
                ])
            ]),
            OpTypePattern("Mul", name="input2"),
        ]),
    ]
    for pattern in patterns:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            input2 = match.get_op('input2')
            input3 = match.get_op('input3')
            if input3.is_const():
                ratio = input3.get_tensor_value()
            else:
                # If the ratio isn't constant, set it to 0
                logger.warning("Dropout node has non-constant ratio. Using ratio=0.0")
                ratio = 0.0
            if input2.inputs[0].type == "RealDiv":
                data = input2.input[1]
            else:
                data = input2.input[0]
            # TODO(tomwildenhain): replace dropout node with identity if ratio is 0
            outputs = match.get_op('outputs')
            op_name = utils.make_name("Dropout")
            out_name = utils.port_name(op_name)
            new_node = g.make_node(
                "Dropout",
                [data],
                outputs=[out_name],
                name=op_name,
                attr={"ratio": ratio},
                shapes=[g.get_shape(input2.input[0])],
                dtypes=[g.get_dtype(input2.input[0])]
            )
            g.replace_all_inputs(ops, outputs.output[0], new_node.output[0])
            nodes_to_remove = []
            for node in match.get_nodes():
                if node.name != input3.name:
                    nodes_to_remove.append(node)
            if g.safe_to_remove_nodes(nodes_to_remove):
                for n in nodes_to_remove:
                    g.remove_node(n.name)
            else:
                logger.warning("Nodes replaced by dropout node cannot be removed because intermediate results are "
                               "referenced elsewhere in graph")

    return ops
