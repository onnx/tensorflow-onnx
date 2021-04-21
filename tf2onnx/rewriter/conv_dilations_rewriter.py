# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.conv_dilations_rewriter - Rewrites the patten used to represent dilations
pat = SpaceToBatchND->DepthwiseConv2dNative->BatchToSpaceND
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

# pylint: disable=invalid-name,unused-argument,missing-docstring, unused-variable


def rewrite_conv_dilations(g, ops):
    pattern1 = \
        OpTypePattern("BatchToSpaceND", name="batch_to_space", inputs=[
            OpTypePattern("DepthwiseConv2dNative|Conv2D|Conv3D", name="conv", inputs=[
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
    pattern2 = \
        OpTypePattern("BatchToSpaceND", name="batch_to_space", inputs=[
            OpTypePattern("Squeeze", name="squeeze", inputs=[
                OpTypePattern("DepthwiseConv2dNative|Conv2D|Conv3D", name="conv", inputs=[
                    OpTypePattern("ExpandDims", name="expand", inputs=[
                        OpTypePattern("SpaceToBatchND", name="space_to_batch", inputs=[
                            OpTypePattern("*"),
                            OpTypePattern("Const|ConstV2"),
                            OpTypePattern("Const|ConstV2"),
                        ]),
                        OpTypePattern("Const|ConstV2"),
                    ]),
                    OpTypePattern("*"),
                ]),
            ]),
            OpTypePattern("Const|ConstV2"),
            OpTypePattern("Const|ConstV2"),
        ])

    for pattern in [pattern1, pattern2]:
        matcher = GraphMatcher(pattern, allow_reorder=False)
        match_results = list(matcher.match_ops(ops))
        for match_result in match_results:
            is_conv_1d = pattern is pattern2
            space_to_batch = match_result.get_op("space_to_batch")
            conv = match_result.get_op("conv")
            batch_to_space = match_result.get_op("batch_to_space")
            if is_conv_1d:
                expand = match_result.get_op("expand")
                expand_axis = expand.inputs[1].get_tensor_value(as_list=True)
                squeeze = match_result.get_op("squeeze")
                squeeze_axes = squeeze.get_attr_value("squeeze_dims")
                if expand_axis not in [1, -3] or squeeze_axes not in [[1], [-3]]:
                    continue

            block_shape1 = space_to_batch.inputs[1].get_tensor_value(as_list=True)
            paddings = space_to_batch.inputs[2].get_tensor_value(as_list=True)
            block_shape2 = batch_to_space.inputs[1].get_tensor_value(as_list=True)
            crops = batch_to_space.inputs[2].get_tensor_value(as_list=True)

            if block_shape1 != block_shape2:
                continue
            ndims = 2 if is_conv_1d else len(block_shape1)
            data_format = b"NHWC" if ndims == 2 else b"NDHWC"
            ones = [1] * (ndims + 2)
            if conv.get_attr_value("dilations", ones) != ones:
                continue
            if conv.get_attr_value("strides", ones) != ones:
                continue
            if conv.get_attr_value("data_format", data_format) != data_format:
                continue
            if conv.get_attr_value("padding") != b"VALID":
                continue


            base_start_pad = [p[0] for p in paddings]
            if any(c[0] != 0 for c in crops):
                continue
            base_end_pad = [p[1] - c[1] for p, c in zip(paddings, crops)]
            if not all(0 <= p[1] - bp < bs for p, bp, bs in zip(paddings, base_end_pad, block_shape1)):
                continue

            if is_conv_1d:
                inp = space_to_batch.input[0]
                g.replace_inputs(expand, [inp, expand.input[1]])
                g.copy_shape(batch_to_space.output[0], squeeze.output[0])
                g.replace_all_inputs(batch_to_space.output[0], squeeze.output[0])
                squeeze_out_shape = g.get_shape(squeeze.output[0])
                g.set_shape(squeeze.input[0], squeeze_out_shape[:1] + [1] + squeeze_out_shape[1:])
                expand_inp_shape = g.get_shape(expand.input[0])
                g.set_shape(expand.output[0], expand_inp_shape[:1] + [1] + expand_inp_shape[1:])

                base_start_pad = [0] + base_start_pad
                base_end_pad = [0] + base_end_pad
                block_shape1 = [1] + block_shape1
            else:
                inp = space_to_batch.input[0]
                kernel = conv.input[1]
                g.replace_inputs(conv, [inp, kernel])
                g.copy_shape(batch_to_space.output[0], conv.output[0])
                g.replace_all_inputs(batch_to_space.output[0], conv.output[0])

            base_pad_flat = [0, 0] + [x for s, e in zip(base_start_pad, base_end_pad) for x in [s, e]] + [0, 0]
            conv.set_attr("dilations", [1] + block_shape1 + [1])
            conv.set_attr("explicit_paddings", base_pad_flat)
            conv.set_attr("padding", "EXPLICIT")

    return g.get_nodes()
