# SPDX-License-Identifier: Apache-2.0


"""
tfl_math
"""

import logging
import numpy as np
from tf2onnx.handler import tfl_op
from tf2onnx import utils

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


def separate_fused_activation_function(ctx, node):
    activation_fn = node.attr['fused_activation_function'].s
    del node.attr['fused_activation_function']
    if activation_fn == b'RELU':
        ctx.insert_new_node_on_output("Relu", node.output[0])
    elif activation_fn == b'RELU6':
        # This is a TF op. We will convert it on the 2nd pass.
        shape = ctx.get_shape(node.output[0])
        dtype = ctx.get_dtype(node.output[0])
        new_node = ctx.make_node("Relu6", [node.output[0]], skip_conversion=False, shapes=[shape], dtypes=[dtype])
        ctx.insert_node_on_output(new_node, node.output[0])
    elif activation_fn == b'TANH':
        ctx.insert_new_node_on_output("Tanh", node.output[0])
    else:
        # TODO: SIGN_BIT and RELU_N1_TO_1 not supported yet
        utils.make_sure(activation_fn == b'NONE', "Unsupported fused activation function %s on node %s",
                        activation_fn, node.name)

@tfl_op(["TFL_ADD"], tf_op="Add")
class TflAdd:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_SUB"], tf_op="Sub")
class TflSub:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_MUL"], tf_op="Mul")
class TflMul:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_DIV"], tf_op="Div")
class TflDiv:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)

@tfl_op(["TFL_LOGISTIC"], tf_op="Sigmoid")
class TflLogistic:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_REDUCE_MAX"], tf_op="Max")
@tfl_op(["TFL_REDUCE_ANY"], tf_op="Any")
@tfl_op(["TFL_REDUCE_PROD"], tf_op="Prod")
class TflReduceOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_LOCAL_RESPONSE_NORMALIZATION"], tf_op="LRN")
class TFlLocalResponseNormalizationOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["depth_radius"] = node.attr["radius"]
        del node.attr["radius"]

@tfl_op(["TFL_RANGE"], tf_op="Range")
class TflRangeOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.set_attr("Tidx", ctx.get_dtype(node.output[0]))

@tfl_op(["TFL_QUANTIZE"], onnx_op="QuantizeLinear")
class TflQuantizeOp:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        scale = node.get_attr_value('scale')
        zero_point = node.get_attr_value('zero_point')
        axis = node.get_attr_value('quantized_dimension')
        np_q_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.output[0]))
        if len(scale) > 1 or len(zero_point) > 1:
            node.set_attr("axis", axis)
        scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale[0], dtype=np.float32))
        zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point[0], dtype=np_q_type))
        ctx.replace_inputs(node, [node.input[0], scale_node.output[0], zero_point_node.output[0]])
        del node.attr["scale"]
        del node.attr["zero_point"]
        del node.attr["quantized_dimension"]

@tfl_op(["TFL_DEQUANTIZE"], onnx_op="DequantizeLinear")
class TflDequantizeOp:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        scale = node.get_attr_value('scale')
        zero_point = node.get_attr_value('zero_point')
        axis = node.get_attr_value('quantized_dimension')
        np_q_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[0]))
        if len(scale) > 1 or len(zero_point) > 1:
            utils.make_sure(ctx.opset >= 13, "Opset 13 is required for per-axis quantization")
            node.set_attr("axis", axis)
            scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale, dtype=np.float32))
            zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point, dtype=np_q_type))
        else:
            scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale[0], dtype=np.float32))
            zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point[0], dtype=np_q_type))
        ctx.replace_inputs(node, [node.input[0], scale_node.output[0], zero_point_node.output[0]])
        del node.attr["scale"]
        del node.attr["zero_point"]
        del node.attr["quantized_dimension"]

def dynamic_quantize_inputs(ctx, node):
    if ctx.opset < 11:
        logger.warning("Opset 11 is required for asymmetric_quantize_inputs of node %s", node.name)
        return
    for i in range(len(node.input)):
        # Don't quantize inputs that are already quantized
        if node.inputs[i].type in ["DequantizeLinear", "TFL_DEQUANTIZE"]:
            continue
        dyn_quant = ctx.make_node("DynamicQuantizeLinear", [node.input[i]], output_count=3, op_name_scope=node.name)
        dyn_quant.skip_conversion = True
        dequant = ctx.make_node("DequantizeLinear", dyn_quant.output, op_name_scope=node.name)
        dequant.skip_conversion = True
        ctx.replace_input(node, node.input[i], dequant.output[0], input_index=i)

@tfl_op(["TFL_FULLY_CONNECTED"])
class TflFullyConnectedOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        separate_fused_activation_function(ctx, node)
        utils.make_sure(node.attr['weights_format'].s == b'DEFAULT',
                        "Only default weights format supported for fully connected op")
        utils.make_sure(node.attr['keep_num_dims'].i == 0,
                        "Only keep_num_dims=False supported for fully connected op")
        if node.attr['asymmetric_quantize_inputs'].i == 1:
            dynamic_quantize_inputs(ctx, node)

        transpose_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[1],
                                                      name=None, input_index=1, perm=[1, 0])
        transpose_node.skip_conversion = True
        node.set_attr("transpose_a", 0)
        node.set_attr("transpose_b", 0)
        node.type = "MatMul"

        if len(node.input) == 3:
            # FIXME: Add a test for this
            bias_inp = node.input[2]
            ctx.replace_inputs(node, node.input[:2])
            add_node = ctx.insert_new_node_on_output("Add", node.output[0], inputs=[node.output[0], bias_inp])
            add_node.skip_conversion = True

        del node.attr["weights_format"]
        del node.attr["keep_num_dims"]
        del node.attr["asymmetric_quantize_inputs"]

@tfl_op(["TFL_SOFTMAX"], tf_op="Softmax")
class TFlSoftmaxOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        beta = node.get_attr_value("beta")
        beta_node = ctx.make_const(utils.make_name("beta"), np.array(beta, dtype=np.float32))
        mul_node = ctx.insert_new_node_on_output("Mul", node.output[0], name=utils.make_name(node.name))
        ctx.replace_inputs(mul_node, [node.output[0], beta_node.output[0]])
