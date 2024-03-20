# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.extract_image_patches_rewriter - Rewrites ExtractImagePatches into supported operations.
"""

import numpy as np
from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring

def rewrite_extract_image_patches(g, ops):
    pattern = OpTypePattern("ExtractImagePatches", name="extract_image_patches")
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match_result in match_results:
        operation = match_result.get_op("extract_image_patches")
        input_shape = g.get_shape(operation.input[0])
        output_shape = operation.output_shapes[0]

        sizes = operation.get_attr_value("ksizes")
        strides = operation.get_attr_value("strides")
        rates = operation.get_attr_value("rates")
        padding = operation.get_attr_str("padding")

        # Our constraints.
        utils.make_sure(0 not in output_shape, "Empty ExtractImagePatches output is unsupported.")
        [_, size_rows, size_cols, _] = sizes

        # Transform input into [N * C, H, W, 1].
        transformed_input = g.make_node("Reshape", inputs=[
            g.make_node("Transpose", inputs=operation.input, attr=dict(perm=[0, 3, 1, 2])).output[0],
            g.make_const(utils.make_name("new_shape"), np.int64([
                input_shape[0] * input_shape[3],
                input_shape[1],
                input_shape[2],
                1,
            ])).output[0],
        ])

        # Create identity kernel.
        k = size_rows * size_cols
        identity_kernel = g.make_node("Reshape", inputs=[
            g.make_node("EyeLike", inputs=[
                g.make_node("ConstantOfShape", inputs=[
                    g.make_const(utils.make_name("eye_size"), np.array([k, k], dtype=np.int64)).output[0],
                ]).output[0],
            ]).output[0],
            g.make_const(utils.make_name("new_shape"), np.array([
                size_rows,
                size_cols,
                1,
                k,
            ], dtype=np.int64)).output[0],
        ])

        # Convolve into [N * C, ?H, ?W, K].
        convolution = g.make_node("Conv2D", inputs=[transformed_input.output[0], identity_kernel.output[0]],
                                  attr=dict(strides=strides, dilations=rates, padding=padding, data_format="NHWC"),
                                  shapes=[[input_shape[0] * input_shape[3], output_shape[1], output_shape[2], k]],
                                  dtypes=operation.output_dtypes, skip_conversion=False)

        # Transform into [N, ?H, ?W, C * K].
        output_node = g.make_node("Reshape", inputs=[
            g.make_node("Transpose", inputs=[
                g.make_node("Reshape", inputs=[
                    convolution.output[0],
                    g.make_const(utils.make_name("new_shape"), np.array([
                        input_shape[0],
                        input_shape[3],
                        output_shape[1],
                        output_shape[2],
                        k,
                    ], dtype=np.int64)).output[0],
                ]).output[0],
            ], attr=dict(perm=[0, 2, 3, 4, 1])).output[0],
            g.make_const(utils.make_name("new_shape"), np.array(output_shape, dtype=np.int64)).output[0],
        ])

        # Replace node.
        g.replace_all_inputs(operation.output[0], output_node.output[0])
        g.remove_node(operation.name)
    return g.get_nodes()
