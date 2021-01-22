# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tfl_math
"""

import logging
import numpy as np
from tf2onnx.handler import tfl_op
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder
from onnx.onnx_pb import TensorProto

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

@tfl_op(["TFL_TFLite_Detection_PostProcess"])
class TflDetectionPostProcess:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # box_encodings.shape = [batch_dim, box_num, 4]
        # class_predictions.shape = [batch_dim, box_num, classes_num(+1)]
        # anchors.shape = [box_num, 4]
        box_encodings, class_predictions, anchors = node.input

        max_int64 = int(utils.get_max_value(np.int64))
        num_classes = node.get_attr_value('num_classes')
        max_detections = node.get_attr_value('max_detections')
        class_predictions = GraphBuilder(ctx).make_slice({'data': class_predictions, 'starts': [-num_classes], 'ends': [max_int64], 'axes': [2]})

        scaling_vector = [node.get_attr_value(a) for a in ['y_scale', 'x_scale', 'h_scale', 'w_scale']]
        scale_const = ctx.make_const(utils.make_name('scale_const'), np.array(scaling_vector, np.float32)).output[0]

        scaled_boxes = ctx.make_node('Div', [box_encodings, scale_const]).output[0]
        anchors_yx = GraphBuilder(ctx).make_slice({'data': anchors, 'starts': [0], 'ends': [2], 'axes': [1]})
        anchors_hw = GraphBuilder(ctx).make_slice({'data': anchors, 'starts': [2], 'ends': [4], 'axes': [1]})
        boxes_yx = GraphBuilder(ctx).make_slice({'data': scaled_boxes, 'starts': [0], 'ends': [2], 'axes': [2]})
        boxes_hw = GraphBuilder(ctx).make_slice({'data': scaled_boxes, 'starts': [2], 'ends': [4], 'axes': [2]})

        scaled_boxes_yx = ctx.make_node('Mul', [boxes_yx, anchors_hw]).output[0]
        boxes_hw_exp = ctx.make_node('Exp', [boxes_hw]).output[0]
        scaled_boxes_hw = ctx.make_node('Mul', [boxes_hw_exp, anchors_hw]).output[0]
        const_half = ctx.make_const(utils.make_name('const_half'), np.array(0.5, np.float32)).output[0]
        boxes_half_hw = ctx.make_node('Mul', [scaled_boxes_hw, const_half]).output[0]
        boxes_center_yx = ctx.make_node('Add', [scaled_boxes_yx, anchors_yx]).output[0]

        boxes_lower_left = ctx.make_node('Sub', [boxes_center_yx, boxes_half_hw]).output[0]
        boxes_upper_right = ctx.make_node('Add', [boxes_center_yx, boxes_half_hw]).output[0]
        adjusted_boxes = ctx.make_node('Concat', [boxes_lower_left, boxes_upper_right], attr={'axis': 2}).output[0]

        iou_threshold = np.array(node.get_attr_value('nms_iou_threshold'), np.float32)
        iou_threshold_const = ctx.make_const(utils.make_name('iou_threshold'), iou_threshold).output[0]

        score_threshold = np.array(node.get_attr_value('nms_score_threshold'), np.float32)
        score_threshold_const = ctx.make_const(utils.make_name('score_threshold'), score_threshold).output[0]

        boxes_per_class = np.array(node.get_attr_value('detections_per_class', 100), np.int64)
        max_boxes_per_class_const = ctx.make_const(utils.make_name('max_boxes_per_class'), boxes_per_class).output[0]

        # scores.shape = [batch_dim, classes_num, box_num]
        scores = ctx.make_node('Transpose', [class_predictions], attr={'perm': [0, 2, 1]}).output[0]

        nms_inputs = [adjusted_boxes, scores, max_boxes_per_class_const, iou_threshold_const, score_threshold_const]
        # batch_index, class_index, box_index
        selected_indices = ctx.make_node('NonMaxSuppression', nms_inputs, attr={'center_point_box': 0}).output[0]

        selected_boxes_idx = GraphBuilder(ctx).make_slice(
            {'data': selected_indices, 'starts': [2], 'ends': [3], 'axes': [1]})
        selected_boxes_idx_sq = GraphBuilder(ctx).make_squeeze({'data': selected_boxes_idx, 'axes': [1]})

        selected_classes = GraphBuilder(ctx).make_slice(
            {'data': selected_indices, 'starts': [1], 'ends': [2], 'axes': [1]})
        selected_classes_sq = GraphBuilder(ctx).make_squeeze({'data': selected_classes, 'axes': [1]})

        box_and_class_idx = ctx.make_node('Concat', [selected_boxes_idx, selected_classes], attr={'axis': 1}).output[0]

        box_cnt = ctx.make_node('Shape', [selected_classes_sq]).output[0]
        box_cnt_float = ctx.make_node('Cast', [box_cnt], attr={'to': TensorProto.FLOAT}).output[0]

        selected_classes_unsq = GraphBuilder(ctx).make_unsqueeze({'data': selected_classes_sq, 'axes': [0]})
        classes_float = ctx.make_node('Cast', [selected_classes_unsq], attr={'to': TensorProto.FLOAT}).output[0]

        adjusted_boxes_sq = GraphBuilder(ctx).make_squeeze({'data': adjusted_boxes, 'axes': [0]})
        detection_boxes = ctx.make_node('Gather', [adjusted_boxes_sq, selected_boxes_idx_sq]).output[0]
        class_predictions_sq = GraphBuilder(ctx).make_squeeze({'data': class_predictions, 'axes': [0]})
        detection_scores = ctx.make_node('GatherND', [class_predictions_sq, box_and_class_idx]).output[0]
        detection_scores_unsq = GraphBuilder(ctx).make_unsqueeze({'data': detection_scores, 'axes': [0]})

        k_const = ctx.make_const(utils.make_name('const_k'), np.array([max_detections], np.int64)).output[0]
        min_k = ctx.make_node('Min', [k_const, box_cnt]).output[0]
        scores_top_k, scores_top_k_idx = ctx.make_node('TopK', [detection_scores, min_k], output_count=2).output

        scores_top_k_idx_unsq = GraphBuilder(ctx).make_unsqueeze({'data': scores_top_k_idx, 'axes': [0]})
        scores_top_k_unsq = GraphBuilder(ctx).make_unsqueeze({'data': scores_top_k, 'axes': [0]})

        selected_classes_sort = ctx.make_node('Gather', [selected_classes_sq, scores_top_k_idx_unsq]).output[0]
        classes_sort_cast = ctx.make_node('Cast', [selected_classes_sort], attr={'to': TensorProto.FLOAT}).output[0]
        detection_boxes_sorted = ctx.make_node('Gather', [detection_boxes, scores_top_k_idx_unsq]).output[0]

        pad_amount = ctx.make_node('Sub', [k_const, box_cnt]).output[0]
        pad_amount_float = ctx.make_node('Cast', [pad_amount], attr={'to': TensorProto.FLOAT}).output[0]

        quad_zero_const = ctx.make_const(utils.make_name('quad_zero_const'), np.array([0, 0, 0, 0], np.int64)).output[0]
        duo_zero_const = ctx.make_const(utils.make_name('duo_zero_const'), np.array([0, 0], np.int64)).output[0]
        zero_const = ctx.make_const(utils.make_name('zero_const'), np.array([0], np.int64)).output[0]

        pads_3d = ctx.make_node('Concat', [quad_zero_const, pad_amount, zero_const], attr={'axis': 0}).output[0]
        pads_2d = ctx.make_node('Concat', [duo_zero_const, zero_const, pad_amount], attr={'axis': 0}).output[0]

        detection_boxes_padded = ctx.make_node('Pad', [detection_boxes_sorted, pads_3d]).output[0]
        detection_classes_padded = ctx.make_node('Pad', [classes_sort_cast, pads_2d]).output[0]
        detection_scores_padded = ctx.make_node('Pad', [scores_top_k_unsq, pads_2d]).output[0]

        #transpose = ctx.make_node('Transpose', [selected_indices])

        # detection_boxes.shape = [batch_dim, cnt, 4]
        # detection_classes.shape = [batch_dim, cnt]
        # detection_scores.shape = [batch_dim, cnt]
        # num_detections.shape = [batch_dim]
        #detection_boxes, detection_classes, detection_scores, num_detections = node.output

        ctx.replace_all_inputs(node.output[0], detection_boxes_padded)
        ctx.replace_all_inputs(node.output[1], detection_classes_padded)
        ctx.replace_all_inputs(node.output[2], detection_scores_padded)
        ctx.replace_all_inputs(node.output[3], box_cnt_float)



        print("Hello")
        pass