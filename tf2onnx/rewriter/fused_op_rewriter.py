# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.fused_op_rewriter - rewrite tensorflow _Fused ops from grappler into other tf ops
"""


# pylint: disable=missing-docstring


def rewrite_fused_ops(g, ops):
    for node in ops:
        if node.type in ["_FusedConv2D", "_FusedMatMul", "_FusedDepthwiseConv2dNative"]:
            op_types = [op.decode() for op in node.get_attr_value("fused_ops")]
            extra_inputs = node.input[2:]
            g.replace_inputs(node, node.input[:2])
            last_output = node.output[0]
            node.type = node.type.replace("_Fused", "")
            dtype = g.get_dtype(node.output[0])
            shape = g.get_shape(node.output[0])
            first_node = None
            for op in op_types:
                new_node = g.make_node(op, [last_output] + extra_inputs, skip_conversion=False,
                                       op_name_scope=node.name, dtypes=[dtype], shapes=[shape])
                last_output = new_node.output[0]
                if first_node is None:
                    first_node = new_node
                extra_inputs = []

            consumers = [n for n in g.find_output_consumers(node.output[0]) if n != first_node]
            g.replace_all_inputs(node.output[0], last_output, consumers)

    return g.get_nodes()
