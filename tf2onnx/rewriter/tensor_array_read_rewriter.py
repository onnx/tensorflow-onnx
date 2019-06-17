# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.tensor_array_read_rewriter - rewrite TensorArrayReadV3 outside while_loop
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_tensor_array_read(g, ops):
    pattern = \
        OpTypePattern("TensorArrayReadV3", name="tensor_array_read", inputs=[
            OpTypePattern("TensorArrayV3"),
            OpTypePattern("*"),
            OpTypePattern("TensorArrayScatterV3", name="tensor_array_scatter", inputs=[
                OpTypePattern("TensorArrayV3"),
                OpTypePattern("*"),
                OpTypePattern("*"),
                OpTypePattern("TensorArrayV3")
            ])
        ])

    matcher = GraphMatcher(pattern, allow_reorder=False)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        ta_read_node = match.get_op("tensor_array_read")
        ta_scatter_node = match.get_op("tensor_array_scatter")
        output = ta_read_node.output[0]
        index = ta_read_node.input[1]
        value = ta_scatter_node.input[2]

        output_shapes = ta_read_node.output_shapes
        output_dtypes = ta_read_node.output_dtypes
        g.remove_node(ta_read_node.name)

        # replace by Gather
        unsqueeze_index = g.make_node("Unsqueeze", [index], attr={"axes": [0]})
        gather = g.make_node("Gather", [value, unsqueeze_index.output[0]])
        g.make_node(
            "Squeeze", [gather.output[0]], outputs=[output], attr={"axes": [0]},
            shapes=output_shapes, dtypes=output_dtypes, op_name_scope=ta_read_node.name,
        )

    return ops
