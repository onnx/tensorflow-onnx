# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.depthwise_conv_dilations_rewriter - Rewrites the patten used to represent dilations
pat = SpaceToBatchND->DepthwiseConv2dNative->BatchToSpaceND
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

# pylint: disable=invalid-name,unused-argument,missing-docstring, unused-variable


def rewrite_depthwise_conv_dilations(g, ops):
    pattern1 = \
        OpTypePattern("BatchToSpaceND", name="batch_to_space", inputs=[
            OpTypePattern("DepthwiseConv2dNative", name="depthwise_conv", inputs=[
                OpTypePattern("SpaceToBatchND", name="space_to_batch", inputs=[
                    OpTypePattern("*"),
                    OpTypePattern("Const|ConstV2"),
                    OpTypePattern("Const|ConstV2"),
                ]),
                OpTypePattern("*"),
            ]),
            OpTypePattern("Const|ConstV2"),
            OpTypePattern("Const|ConstV2"),
        ])

    for pattern in [pattern1]:
        matcher = GraphMatcher(pattern, allow_reorder=False)
        match_results = list(matcher.match_ops(ops))
        for match_result in match_results:
            space_to_batch = match_result.get_op("space_to_batch")
            depthwise_conv = match_result.get_op("depthwise_conv")
            batch_to_space = match_result.get_op("batch_to_space")

            block_shape1 = space_to_batch.inputs[1].get_tensor_value(as_list=True)
            paddings = space_to_batch.inputs[2].get_tensor_value(as_list=False).flatten().tolist()
            block_shape2 = batch_to_space.inputs[1].get_tensor_value(as_list=True)
            crops = batch_to_space.inputs[2].get_tensor_value(as_list=True)
            if block_shape1 != block_shape2:
                continue
            if depthwise_conv.get_attr_value("dilations", [1, 1, 1, 1]) != [1, 1, 1, 1]:
                continue
            if depthwise_conv.get_attr_value("strides", [1, 1, 1, 1]) != [1, 1, 1, 1]:
                continue
            if depthwise_conv.get_attr_value("data_format", b"NHWC") != b"NHWC":
                continue
            if depthwise_conv.get_attr_value("padding") != b"VALID":
                continue
            if crops != [[0, 0], [0, 0]]:
                continue

            inp = space_to_batch.input[0]
            kernel = depthwise_conv.input[1]

            g.replace_inputs(depthwise_conv, [inp, kernel])
            depthwise_conv.set_attr("dilations", [1] + block_shape1 + [1])
            depthwise_conv.set_attr("explicit_paddings", [0, 0] + paddings + [0, 0])
            depthwise_conv.set_attr("padding", "EXPLICIT")
            g.copy_shape(batch_to_space.output[0], depthwise_conv.output[0])
            g.replace_all_inputs(batch_to_space.output[0], depthwise_conv.output[0])

    return g.get_nodes()
