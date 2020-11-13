
from tf2onnx.handler import tfl_op
from tf2onnx import constants, utils
import numpy as np

@tfl_op(["TFL_ADD"], tf_op="Add")
class TflAdd:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")

@tfl_op(["TFL_SUB"], tf_op="Sub")
class TflSub:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")

@tfl_op(["TFL_MUL"], tf_op="Mul")
class TflMul:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")

@tfl_op(["TFL_DIV"], tf_op="Div")
class TflDiv:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")

@tfl_op(["TFL_CONCATENATION"], onnx_op="Concat")
class TflConcatenation:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_SPLIT"], tf_op="Split")
class TflSplit:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr['num_split'] = node.attr['num_splits']
        del node.attr['num_splits']

@tfl_op(["TFL_SPLIT_V"], tf_op="SplitV")
class TflSplit:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr['num_split'] = node.attr['num_splits']
        del node.attr['num_splits']

direct_tfl_to_tf_map = [
    ("TFL_GREATER", "Greater"),
    ("TFL_GREATER_EQUAL", "GreaterEqual"),
    ("TFL_LESS", "Less"),
    ("TFL_LESS_EQUAL", "LessEqual"),
    ("TFL_EQUAL", "Equal"),
    ("TFL_EXP", "Exp"),
    ("TFL_SQRT", "Sqrt"),
    ("TFL_NEG", "Neg"),
    ("TFL_POW", "Pow"),
    ("TFL_FLOOR", "Floor"),
    ("TFL_CEIL", "Ceil"),
    ("TFL_TANH", "Tanh"),
    ("TFL_SIN", "Sin"),
    ("TFL_COS", "Cos"),
    ("TFL_LOG", "Log"),
    ("TFL_ABS", "Abs"),
    ("TFL_LOGICAL_AND", "LogicalAnd"),
    ("TFL_LOGICAL_NOT", "LogicalNot"),
    ("TFL_LOGICAL_OR", "LogicalOr"),
]
# for tfl_op_name, tf_op_name in direct_tfl_to_tf_map:
#     @tfl_op([tfl_op_name], tf_op=tf_op_name)
#     class TflDirectOp:
#         @classmethod
#         def to_tf(cls, ctx, node, **kwargs):
#             pass

@tfl_op(["TFL_TRANSPOSE"], tf_op="Transpose")
class TflTranspose:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_GATHER"], onnx_op="Gather")
class TflGather:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_RESHAPE"], tf_op="Reshape")
class TflReshape:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        if 'new_shape' in node.attr:
            del node.attr['new_shape']
        #utils.make_sure('new_shape' not in node.attr, "new_shape attr not yet supported for reshape (use input)")

@tfl_op(["TFL_SLICE"], tf_op="Slice")
class TflSlice:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_CAST"], tf_op="Cast")
class TflCast:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        dst = ctx.get_dtype(node.output[0])
        if "out_data_type" in node.attr:
            del node.attr["out_data_type"]
            del node.attr["in_data_type"]
        node.set_attr("to", dst)


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

@tfl_op(["TFL_PACK"], tf_op="Pack")
class TFlPackOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["N"] = node.attr["values_count"]
        del node.attr["values_count"]

@tfl_op(["TFL_LOCAL_RESPONSE_NORMALIZATION"], tf_op="LRN")
class TFlLocalResponseNormalizationOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["depth_radius"] = node.attr["radius"]
        del node.attr["radius"]

@tfl_op(["TFL_PADV2"], tf_op="PadV2")
class TflPadV2Op:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_NON_MAX_SUPPRESSION_V4"], tf_op="NonMaxSuppressionV4")
class TflNonMaxSuppressionV4Op:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.set_attr("pad_to_max_output_size", 0)

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
        np_q_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.output[0]))
        utils.make_sure(len(scale) == 1, "Quantize only supports scalar vals")
        utils.make_sure(len(zero_point) == 1, "Quantize only supports scalar vals")
        scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale[0], dtype=np.float32))
        zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point[0], dtype=np_q_type))
        ctx.replace_inputs(node, [node.input[0], scale_node.output[0], zero_point_node.output[0]])
        del node.attr["scale"]
        del node.attr["zero_point"]

@tfl_op(["TFL_DEQUANTIZE"], onnx_op="DequantizeLinear")
class TflDequantizeOp:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        scale = node.get_attr_value('scale')
        zero_point = node.get_attr_value('zero_point')
        np_q_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[0]))
        utils.make_sure(len(scale) == 1, "Quantize only supports scalar vals")
        utils.make_sure(len(zero_point) == 1, "Quantize only supports scalar vals")
        scale_node = ctx.make_const(utils.make_name("scale"), np.array(scale[0], dtype=np.float32))
        zero_point_node = ctx.make_const(utils.make_name("zero_point"), np.array(zero_point[0], dtype=np_q_type))
        ctx.replace_inputs(node, [node.input[0], scale_node.output[0], zero_point_node.output[0]])
        del node.attr["scale"]
        del node.attr["zero_point"]

@tfl_op(["TFL_FULLY_CONNECTED"])
class TflFullyConnectedOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")
        utils.make_sure(node.attr['weights_format'].s == b'DEFAULT', "Only default weights format supported for fully connected op")
        utils.make_sure(len(node.input) == 2, "Only supports 2 inputs for TFL_FULLY_CONNECTED")  # FIXME
        if len(node.input) == 2:
            transpose_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[1], name=None, input_index=1, perm=[1, 0])
            transpose_node.skip_conversion = True
            node.set_attr("transpose_a", 0)
            node.set_attr("transpose_b", 0)
            node.type = "MatMul"
        else:
            assert False

# DONE

@tfl_op(["TFL_UNIQUE"], tf_op="Unique")
class TFlUniqueOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.attr["out_idx"] = node.attr["idx_out_type"]
        del node.attr["idx_out_type"]

@tfl_op(["TFL_TOPK_V2"], tf_op="TopKV2")
class TFlTopKV2Op:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.set_attr("sorted", 1)

@tfl_op(["TFL_SOFTMAX"], tf_op="Softmax")
class TFlSoftmaxOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        beta = node.get_attr_float("beta")
        beta_node = ctx.make_const(utils.make_name("beta"), np.array(beta, dtype=np.float32))
        mul_node = ctx.insert_new_node_on_output("Mul", node.output[0], name=utils.make_name(node.name))
        ctx.replace_inputs(mul_node, [node.output[0], beta_node.output[0]])


@tfl_op(["TFL_SPACE_TO_BATCH_ND"], tf_op="SpaceToBatchND")
class TFlSpaceToBatchNDOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass

@tfl_op(["TFL_SPACE_TO_DEPTH"], tf_op="SpaceToDepth")
class TFlSpaceToDepthOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        node.set_attr("data_format", "NHWC")