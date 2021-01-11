# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter - rewrite tensorflow QuantizeAndDequantizeV2|QuantizeAndDequantizeV3 op
"""

import numpy as np
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx import utils

# pylint: disable=missing-docstring

def extract_numpy_array(node):
    return np.frombuffer(node.attr["value"].t.raw_data, dtype="float32")

def create_qdq_nodes(g, match_results):

    for match in match_results:
        qdq_node = match.get_op('output')
        qdq_node_output_dtype = g.get_dtype(qdq_node.output[0])
        qdq_node_output_shape = g.get_shape(qdq_node.output[0])

        # Get the attributes of qdq node
        narrow_range = qdq_node.attr['narrow_range'].i
        signed_input = qdq_node.attr['signed_input'].i

        min_quantized, max_quantized = [-127, 127]
        if not narrow_range and signed_input:
            min_quantized = -128

        if not signed_input:
            min_quantized, max_quantized = [0, 255]

        # Get axis attribute for per channel implementation.
        if 'axis' in qdq_node.attr:
            axis = qdq_node.attr['axis'].i

        # Get the min and max value of the inputs to QDQ op
        min_value = extract_numpy_array(qdq_node.inputs[1])
        max_value = extract_numpy_array(qdq_node.inputs[2])

        num_channels = min_value.shape[0]
        scales = np.zeros(num_channels, dtype=np.float32)
        zero_point_dtype = np.int8 if signed_input else np.uint8
        zero_point = np.zeros(num_channels, dtype=zero_point_dtype)

        for i in range(num_channels):
            # Calculate scales from the min and max values
            scale_from_min_side = min_quantized/min_value[i] if min_quantized*min_value[i] > 0 else max_quantized
            scale_from_max_side = max_quantized/max_value[i] if max_quantized*max_value[i] > 0 else max_quantized

            if scale_from_min_side < scale_from_max_side:
                scale = scale_from_min_side
            else:
                scale = scale_from_max_side

            utils.make_sure(scale > 0, "Quantize/Dequantize scale must be greater than zero")
            scales[i] = np.float32(scale)

        # Set scalars for scale and zero point for per layer quantization
        if num_channels == 1:
            scales = scales[0]
            zero_point = zero_point[0]
            attrs = {}
        else:
            utils.make_sure(axis and axis != -1, "Axis must be specified for per channel quantization")
            utils.make_sure(g.opset >= 13, "Opset >= 13 is required for per channel quantization")
            attrs = {'axis': axis}

        # Split it into QuantizeLinear and DequantizeLinear and remove the QDQ node reference
        inverse_scale = (1/scales).astype(np.float32)
        y_quant_scale = g.make_const(name=utils.make_name("y_quant_scale"), np_val=inverse_scale)
        y_zero_point = g.make_const(name=utils.make_name("y_zero_point"), np_val=zero_point)
        quant_node = g.make_node(op_type="QuantizeLinear",
                                 inputs=[qdq_node.input[0], y_quant_scale.output[0],
                                         y_zero_point.output[0]],
                                 shapes=[qdq_node_output_shape],
                                 attr=attrs,
                                 dtypes=[qdq_node_output_dtype],
                                 name=utils.make_name("QuantLinearNode"))

        g.set_shape(quant_node.output[0], qdq_node_output_shape)

        g.remove_node(qdq_node.name)

        y_dequant_scale = g.make_const(name=utils.make_name("y_dequant_scale"), np_val=inverse_scale)
        y_inv_zero_point = g.make_const(name=utils.make_name("y_inv_zero_point"), np_val=zero_point)
        dequant_node = g.make_node(op_type="DequantizeLinear",
                                   inputs=[quant_node.output[0], y_dequant_scale.output[0],
                                           y_inv_zero_point.output[0]],
                                   outputs=[qdq_node.output[0]],
                                   shapes=[qdq_node_output_shape],
                                   attr=attrs,
                                   dtypes=[qdq_node_output_dtype],
                                   name=utils.make_name("DequantLinearNode"))
        g.set_shape(dequant_node.output[0], qdq_node_output_shape)

    return g.get_nodes()

def rewrite_quantize_and_dequantize(g, ops):

    pattern_for_qdq_v2 = \
        OpTypePattern('QuantizeAndDequantizeV2', name='output', inputs=[
            OpTypePattern("*"),
            OpTypePattern(None),
            OpTypePattern(None),
        ])
    pattern_for_qdq_v3 = \
        OpTypePattern('QuantizeAndDequantizeV3', name='output', inputs=[
            OpTypePattern("*"),
            OpTypePattern(None),
            OpTypePattern(None),
            OpTypePattern(None),
        ])

    # Match all the patterns for QDQ ops
    patterns = [pattern_for_qdq_v3, pattern_for_qdq_v2]
    match_results = []
    for pattern in patterns:
        matcher = GraphMatcher(pattern)
        results = list(matcher.match_ops(ops))
        match_results.extend(results)

    return create_qdq_nodes(g, match_results)
