# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx random normal op
"""

from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_random_normal(g, ops):
    pattern1 = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', name='input2', inputs=[
                OpTypePattern('RandomStandardNormal', name='input1', inputs=["*"]), "*"
            ]), "*"
        ])

    pattern2 = \
        OpTypePattern('Identity', name='output', inputs=[
            OpTypePattern('Identity', name='input2', inputs=[
                OpTypePattern('RandomStandardNormal', name='input1', inputs=["*"])
            ])
        ])

    pattern_list = [pattern1, pattern2]
    for pattern in pattern_list:
        matcher = GraphMatcher(pattern)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            output = match.get_op('output')
            if output.type == 'Add':
                # pattern 1
                mean = output.inputs[1].get_tensor_value()
            else:
                # pattern 2
                mean = 0.0
            input2 = match.get_op('input2')
            if input2.type == 'Mul':
                scale = input2.inputs[1].get_tensor_value()
            else:
                scale = 1.0
            dtype = g.get_dtype(output.output[0])
            op_name = utils.make_name("RandomNormal")
            out_name = utils.port_name(op_name)

            rn_op = match.get_op('input1')
            seed = float(rn_op.get_attr('seed2').i)

            attr = {"mean": mean, "scale": scale, "dtype": dtype, "seed": seed}
            if rn_op.inputs[0].type == "Shape":
                shape_node = rn_op.inputs[0]
                new_node = g.make_node("RandomNormalLike", [shape_node.input[0]], outputs=[out_name], name=op_name,
                                       attr=attr)
            else:
                shape = g.get_shape(output.output[0])
                if shape is None or -1 in shape:
                    continue
                attr['shape'] = shape
                new_node = g.make_node("RandomNormal", [], outputs=[out_name], name=op_name, attr=attr)

            g.replace_all_inputs(output.output[0], new_node.output[0], ops=ops)
            g.safe_remove_nodes(match.get_nodes())
    return ops
