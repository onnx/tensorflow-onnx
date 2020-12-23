# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
reduction
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import onnx_pb, helper

from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

@tf_op("Min", onnx_op="ReduceMin")
@tf_op("Max", onnx_op="ReduceMax")
@tf_op("Mean", onnx_op="ReduceMean")
@tf_op("Sum", onnx_op="ReduceSum")
@tf_op("Prod", onnx_op="ReduceProd")
class ReduceOpBase:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        axes_node = node.inputs[1]
        axes = axes_node.get_tensor_value()
        if np.isscalar(axes):
            axes = [axes]
        input_shape = ctx.get_shape(node.input[0])
        if input_shape is None:
            if any([val < 0 for val in axes]) and ctx.opset < 11:
                raise ValueError("reduce_op: cannot have negative axis if opset < 11 because we don't know input rank")
        else:
            input_rank = len(ctx.get_shape(node.input[0]))
            axes = [val + input_rank if val < 0 else val for val in axes]

        node.set_attr("axes", axes)
        ctx.remove_input(node, node.input[1], 1)
        keep_dims = node.get_attr_value("keep_dims", 0)
        node.set_attr("keepdims", keep_dims)
        del node.attr['keep_dims']

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        if node.type == "ReduceSum":
            keep_dims = node.get_attr_value("keep_dims", 0)
            node.set_attr("keepdims", keep_dims)
            del node.attr['keep_dims']
            node.set_attr("noop_with_empty_axes", 1)
            if ctx.get_dtype(node.input[1]) != onnx_pb.TensorProto.INT64:
                ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)
            input_shape = ctx.get_shape(node.input[1])
            input_rank = len(input_shape) if input_shape is not None else None
            if input_rank != 1:
                new_shape = ctx.make_const(utils.make_name("reshape_const"), np.array([-1], np.int64))
                ctx.insert_new_node_on_input(node, "Reshape", [node.input[1], new_shape.output[0]])
        else:
            cls.version_11(ctx, node, **kwargs)

@tf_op(["ArgMax", "ArgMin"])
class ArgMax:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
        # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
        axis_node = node.inputs[1]
        axis = axis_node.get_tensor_value()
        if axis < 0:
            # ArgMax|ArgMin in onnx don't necessary support negative axis(not in doc explicitly)
            input_shape = ctx.get_shape(node.input[0])
            dim_count = len(input_shape) if input_shape else 0
            axis = dim_count + axis

        # TF ArgMin/ArgMax may return int32 or int64
        # Onnx ArgMin/ArgMax only supports int64 output, add cast if needed
        if node.get_attr_int("output_type") == onnx_pb.TensorProto.INT32:
            # current node will return int64 after conversion, which differs from previous dtype got from tf
            ctx.set_dtype(node.output[0], onnx_pb.TensorProto.INT64)
            op_name = utils.make_name("Cast")
            cast_node = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name,
                                                      to=onnx_pb.TensorProto.INT32)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT32)
            ctx.copy_shape(node.output[0], cast_node.output[0])

        node.set_attr("axis", axis)
        node.set_attr("keepdims", 0)
        ctx.remove_input(node, node.input[1], 1)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic same
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        # Opset 12 adds extra attribute 'select_last_index'
        # No changes needed
        cls.version_1(ctx, node, **kwargs)

@tf_op(["All", "Any"])
class AllAny:
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        # T output = All(T x, list(int) reduce_indices, @bool keepdims)
        # T output = Any(T x, list(int) reduce_indices, @bool keepdims)
        reduce_dim = node.inputs[1].get_tensor_value()

        # for Any, the reduce_indices can be scalar as observed.
        if np.isscalar(reduce_dim):
            reduce_dim = [reduce_dim]

        if ctx.opset < 11:
            utils.make_sure(all(i >= 0 for i in reduce_dim), "negative reduce axis is not supported in onnx for now")

        cast = ctx.make_node(op_type="Cast", inputs=[node.input[0]], attr={"to": onnx_pb.TensorProto.FLOAT})
        keepdims = helper.get_attribute_value(node.get_attr("keep_dims"))
        op_type = "ReduceMin" if node.type == "All" else "ReduceSum"

        if op_type == "ReduceSum":
            reduce_node_output = GraphBuilder(ctx).make_reduce_sum(
                {"data": cast.output[0], "axes": reduce_dim, "keepdims": keepdims, "noop_with_empty_axes": 1})
        else:
            reduce_node_output = ctx.make_node(op_type=op_type, inputs=cast.output,
                                               attr={"axes": reduce_dim, "keepdims": keepdims}).output[0]

        zero_node = ctx.make_const(utils.make_name("zero_reduce"), np.array(0, dtype=np.float32))

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node(op_type="Greater", inputs=[reduce_node_output, zero_node.output[0]],
                      name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


@tf_op("AddN")
class AddN():
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        node.type = "Sum"


@tf_op(["SegmentSum", "SegmentProd", "SegmentMax", "SegmentMin", "SegmentMean",
        "SparseSegmentSum", "SparseSegmentMean", "SparseSegmentSqrtN",
        "SparseSegmentSumWithNumSegments", "SparseSegmentMeanWithNumSegments", "SparseSegmentSqrtNWithNumSegments",
        "UnsortedSegmentSum", "UnsortedSegmentProd", "UnsortedSegmentMax", "UnsortedSegmentMin"])
class SegmentSum():
    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        node_inputs = node.input
        num_segments_specified = False
        if node.type.endswith("WithNumSegments") or node.type.startswith("Unsorted"):
            num_segments_specified = True
            num_segments = node_inputs.pop()
            node.type = node.type.replace("WithNumSegments", "")
            node.type = node.type.replace("Unsorted", "")
        if node.type.startswith("Sparse"):
            data_inp, indices_inp, segment_inp = node_inputs
            gather_node = ctx.make_node("Gather", [data_inp, indices_inp], attr={'axis': 0})
            data_inp = gather_node.output[0]
            node.type = node.type.replace("Sparse", "")
        else:
            data_inp, segment_inp = node_inputs

        # Data has shape [n, a, b, ..., c]
        data_shape = ctx.get_shape(data_inp)
        data_rank = len(data_shape) if data_shape is not None else None
        data_dtype = ctx.get_dtype(data_inp)
        data_np_dtype = utils.map_onnx_to_numpy_type(data_dtype)
        seg_np_dtype = utils.map_onnx_to_numpy_type(ctx.get_dtype(segment_inp))

        if num_segments_specified and ctx.get_dtype(segment_inp) != ctx.get_dtype(num_segments):
            num_segments = ctx.make_node("Cast", [num_segments], attr={"to": ctx.get_dtype(segment_inp)}).output[0]

        data_is_float = np.dtype(data_np_dtype).kind == 'f'
        data_is_int = np.dtype(data_np_dtype).kind == 'i'
        utils.make_sure(data_is_float or data_is_int, "dtype for Segment ops must be float or int")

        if node.type in ["SegmentSum", "SegmentMean", "SegmentSqrtN"]:
            onnx_op = "ReduceSum"
            identity_value = np.array(0, dtype=data_np_dtype)
        elif node.type == "SegmentProd":
            onnx_op = "ReduceProd"
            identity_value = np.array(1, dtype=data_np_dtype)
        elif node.type == "SegmentMax":
            onnx_op = "ReduceMax"
            if data_is_float:
                identity_value = np.array('-inf', dtype=data_np_dtype)
            else:
                identity_value = np.iinfo(data_np_dtype).min
        elif node.type == "SegmentMin":
            onnx_op = "ReduceMin"
            if data_is_float:
                identity_value = np.array('inf', dtype=data_np_dtype)
            else:
                identity_value = np.iinfo(data_np_dtype).max

        if not num_segments_specified:
            max_segment = ctx.make_node("ReduceMax", [segment_inp], attr={'axes': [0], 'keepdims': 0})
            one_const = ctx.make_const(utils.make_name("const_one"), np.array(1, dtype=seg_np_dtype))
            num_segments = ctx.make_node("Add", [max_segment.output[0], one_const.output[0]]).output[0]
        # ORT doesn't support bool for OneHot so we use float32 and cast to bool
        onehot_values = ctx.make_const(utils.make_name("onehot_values"), np.array([0, 1], dtype=np.float32))
        # one_hot_node has shape [s, n] (s is # segments)
        one_hot_node = ctx.make_node("OneHot", [segment_inp, num_segments, onehot_values.output[0]],
                                     attr={'axis': 0})
        if node.type == "SegmentMean":
            scaling_node_output = GraphBuilder(ctx).make_reduce_sum(
                {"data": one_hot_node.output[0], "axes": [1], "keepdims": 0, "noop_with_empty_axes": 1})
        elif node.type == "SegmentSqrtN":
            seg_cnts_node_output = GraphBuilder(ctx).make_reduce_sum(
                {"data": one_hot_node.output[0], "axes": [1], "keepdims": 0, "noop_with_empty_axes": 1})
            scaling_node_output = ctx.make_node("Sqrt", [seg_cnts_node_output]).output[0]
        else:
            scaling_node_output = None

        if scaling_node_output is not None and num_segments_specified:
            # If empty segments are possible, we must avoid division by zero
            const_one_float = ctx.make_const(utils.make_name("const_one_float"), np.array(1, dtype=np.float32))
            scaling_node_output = ctx.make_node("Max", [scaling_node_output, const_one_float.output[0]]).output[0]


        if onnx_op == "ReduceSum":
            # If the op is a summation, we can use MatMul instead of Where, which is faster

            # Data shape is [n, a, b, ..., c]
            data_shape_node = ctx.make_node("Shape", [data_inp])
            new_shape = ctx.make_const(utils.make_name("reshape_const"), np.array([0, -1], dtype=np.int64))
            # Reshape the data from [n, a, b, ..., c] to [n, P]
            data_reshape = ctx.make_node("Reshape", [data_inp, new_shape.output[0]])

            one_hot_cast = one_hot_node
            if data_dtype != onnx_pb.TensorProto.FLOAT:
                one_hot_cast = ctx.make_node("Cast", [one_hot_node.output[0]], attr={'to': data_dtype})

            # Shapes [s, n] * [n, P] => [s, P]
            product = ctx.make_node("MatMul", [one_hot_cast.output[0], data_reshape.output[0]], op_name_scope=node.name)
            if scaling_node_output is not None:
                scaling_node_unsqueeze = GraphBuilder(ctx).make_unsqueeze(
                    {'data': scaling_node_output, 'axes': [1]}, return_node=True)
                product = ctx.make_node("Div", [product.output[0], scaling_node_unsqueeze.output[0]])

            # Create new shape [0, a, b, ..., c]
            max_int64 = int(utils.get_max_value(np.int64))
            new_shape_slice = GraphBuilder(ctx).make_slice(
                {"data": data_shape_node.output[0], "ends": [max_int64], "starts": [1], "axes": [0]})
            zero_const = ctx.make_const(utils.make_name("zero_const"), np.array([0], dtype=np.int64))
            new_shape = ctx.make_node("Concat", [zero_const.output[0], new_shape_slice], attr={'axis': 0})

            shapes = node.output_shapes
            dtypes = node.output_dtypes
            ctx.remove_node(node.name)
            # Reshape result from [s, P] to [s, a, b, ..., c]
            ctx.make_node("Reshape", [product.output[0], new_shape.output[0]],
                          name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
            return

        identity_const = ctx.make_const(utils.make_name("const_identity"), identity_value)
        one_hot_bool = ctx.make_node("Cast", [one_hot_node.output[0]], attr={"to": onnx_pb.TensorProto.BOOL})
        one_hot_unsqueeze = one_hot_bool

        # Make one_hot_unsqueeze have shape [s, n, 1, 1, ..., 1]
        if data_rank is None:
            # Unsqueeze requires known rank, but we can use Reshape if rank is unknown
            shape_node = ctx.make_node("Shape", [data_inp])
            rank_node = ctx.make_node("Shape", [shape_node.output[0]])
            one_const_int64 = ctx.make_const(utils.make_name("const_one"), np.array([1], dtype=np.int64))
            num_unsqueeze_dims = ctx.make_node("Sub", [rank_node.output[0], one_const_int64.output[0]])

            one_tensor = helper.make_tensor("value", onnx_pb.TensorProto.INT64, dims=[1], vals=[1])
            unsqueeze_dims = ctx.make_node("ConstantOfShape", inputs=[num_unsqueeze_dims.output[0]],
                                           attr={"value": one_tensor})
            # Zero indicates a dimension should be unchanged
            double_zero_const = ctx.make_const(utils.make_name("double_zero"), np.array([0, 0], dtype=np.int64))
            expanded_shape = ctx.make_node("Concat", [double_zero_const.output[0], unsqueeze_dims.output[0]],
                                           attr={'axis': 0})
            one_hot_unsqueeze = ctx.make_node("Reshape", [one_hot_bool.output[0], expanded_shape.output[0]])
        elif data_rank > 1:
            new_dims = list(range(2, 2 + data_rank - 1))
            one_hot_unsqueeze = GraphBuilder(ctx).make_unsqueeze(
                {'data': one_hot_bool.output[0], 'axes': new_dims}, return_node=True)

        # Shape of data:       [n, a, b, ..., c]
        # Shape of one_hot: [s, n, 1, 1, ..., 1]
        # Broadcast left-pads shape with 1s, so result is shape: [s, n, a, b, ..., c]
        where_node = ctx.make_node("Where", [one_hot_unsqueeze.output[0], data_inp, identity_const.output[0]])

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        # After reduction over axis 1, shape is: [s, a, b, ..., c]
        ctx.make_node(onnx_op, [where_node.output[0]], attr={'axes': [1], 'keepdims': 0},
                      name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        cls.any_version(9, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        cls.any_version(13, ctx, node, **kwargs)
