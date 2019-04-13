# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tensor
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto

import tf2onnx
from tf2onnx import constants, utils
from tf2onnx.handler import tf_op


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring,unused-variable

def _convert_shapenode_to_int64(ctx, node, input_number):
    """cast int32 shape into int64 shape."""
    name = node.input[input_number]

    cast_node = ctx.insert_new_node_on_input(node, "Cast", name)
    cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
    ctx.copy_shape(name, cast_node.output[0])


def _wrap_concat_with_cast(ctx, node):
    """wrap concat in casts for opset < 8 since it only supports."""
    supported_types = [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16]
    dtype = ctx.get_dtype(node.output[0])
    need_casting = dtype not in supported_types
    if need_casting:
        output_name = node.output[0]
        # cast each inputs to float
        for i, inp in enumerate(node.inputs):
            input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[i])
            input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(input_cast.output[0], onnx_pb.TensorProto.FLOAT)
        next_nodes = ctx.find_output_consumers(node.output[0])
        # cast output back to dtype unless the next op is a cast
        if next_nodes[0].type != "Cast":
            op_name = utils.make_name(node.name)
            output_cast = ctx.insert_new_node_on_output("Cast", output_name, name=op_name)
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(output_name, output_cast.output[0])

@tf_op(["Size", "Flatten", "Dropout"])
class DirectOp:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        pass


@tf_op("Identity")
class Identity:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        if node.inputs[0].is_const():
            # should not remove the identity node if it is output of the graph
            if node.output[0] in ctx.outputs:
                return
            # if identity has a const as input, remove it
            input_name = node.input[0]
            output_name = node.output[0]
            ctx.replace_all_inputs(ctx.get_nodes(), output_name, input_name)
            ctx.remove_node(node.name)


@tf_op("Reshape")
class Reshape:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = Reshape(T tensor, Tshape shape, @type Tshape)
        # T reshaped = Reshape(T data, @INTS shape) - but takes a optional 2nd input for shape
        shape_node = node.inputs[1]
        shape = shape_node.get_tensor_value()
        if shape is None:
            logger.error("Reshape on node %s does not have a const shape", node.name)
            return
        ctx.remove_input(node, node.input[1])
        node.set_attr("shape", shape)
        ctx.set_shape(node.output[0], shape)

    @classmethod
    def version_5(cls, ctx, node, **kwargs):
        dtype = ctx.get_dtype(node.output[0])
        need_casting = dtype in [onnx_pb.TensorProto.INT32,
                                 onnx_pb.TensorProto.INT16,
                                 onnx_pb.TensorProto.INT64]
        # onnx wants reshape.input[1] to have the value be int64 which is not the case for tensorflow.
        _convert_shapenode_to_int64(ctx, node, 1)
        if ctx.opset >= 8 or not need_casting:
            # onnx reshape can handle the type - done
            return

        # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
        input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
        input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.output[0], input_cast.output[0])

        # if the next node is already a cast we don't need to insert another one
        next_nodes = ctx.find_output_consumers(node.output[0])
        if len(next_nodes) != 1 or next_nodes[0].type != "Cast":
            op_name = utils.make_name(node.name)
            output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name)
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(node.output[0], output_cast.output[0])


@tf_op("Squeeze")
class Squeeze:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = Squeeze(T input, @list(int) squeeze_dims)
        # T squeezed = Squeeze(T data, @AttrType.INTS axes), axes are list of positive integers.
        axis = node.get_attr("axis")
        if not axis:
            axis = node.get_attr("squeeze_dims")
            if axis:
                del node.attr["squeeze_dims"]
        else:
            del node.attr["axis"]

        if axis and axis.ints:
            axis = axis.ints
            neg_axis = any([val < 0 for val in axis])
            if neg_axis:
                shape = ctx.get_shape(node.input[0])
                utils.make_sure(shape is not None, "squeeze input shape cannot be None")
                shape_len = len(shape)
                axis = [a + shape_len if a < 0 else a for a in axis]
        else:
            shape = ctx.get_shape(node.input[0])
            utils.make_sure(shape is not None, "squeeze input shape cannot be None")
            axis = [i for i, j in enumerate(shape) if j == 1]
        node.set_attr("axes", axis)


@tf_op("Transpose")
class Transpose:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T y = Transpose(T x, Tperm perm, @type Tperm)
        # T transposed = Transpose(T data, @INTS perm)
        if len(node.input) > 1:
            perm = node.inputs[1]
            if perm.is_const():
                # perms is passed as const
                dims = perm.get_tensor_value()
            else:
                # calculate perms from shape
                shape = ctx.get_shape(node.input[1])
                dims = [i for i in range(len(shape) - 1, -1)]
            ctx.remove_input(node, node.input[1])
            node.set_attr("perm", dims)
        else:
            # graph rewrite moved perm to attribute
            pass


@tf_op("Concat")
class Concat:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # old concat op has axis as input[0]
        node.type = "Concat"
        axis_node = node.inputs[0]
        axis_val = axis_node.get_tensor_value()
        ctx.remove_input(node, node.input[0])

        if axis_val < 0:  # onnxruntime does not support -1 axis, but TF supports.
            input_shape = ctx.get_shape(node.input[0])
            axis_val = len(input_shape) + axis_val
        node.set_attr("axis", axis_val)

        if ctx.opset < 8:
            # opset < 8: might need to wrap concat in casts since only float is supported
            _wrap_concat_with_cast(ctx, node)
            return


@tf_op("ConcatV2")
class ConcatV2:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = ConcatV2(T values, Tidx axis, @int N, @type Tidx)
        # T concat_result = Concat(T inputs, @INT axis)
        # if any input is empty, remove the input and concat the others
        # NOTE: workaround for https://github.com/Microsoft/onnxruntime/issues/681
        node.type = "Concat"
        for i, inp in enumerate(node.inputs):
            if inp.is_const() and inp.get_tensor_value(as_list=False).size == 0:
                ctx.remove_input(node, node.input[i])
        # all inputs are deleted
        if not node.input:
            raise RuntimeError("all inputs of {} are empty".format(name))

        axis_node = node.inputs[-1]
        utils.make_sure(axis_node.is_const(), "{} needs to be const".format(axis_node.name))
        axis_val = axis_node.get_tensor_value()
        ctx.remove_input(node, node.input[-1])

        if axis_val < 0:  # onnxruntime does not support -1 axis, but TF supports.
            input_shape = ctx.get_shape(node.input[0])
            utils.make_sure(input_shape is not None, "shape of {} is None".format(node.input[0]))
            axis_val = len(input_shape) + axis_val
        node.set_attr("axis", axis_val)

        if ctx.opset < 8:
            # opset < 8: might need to wrap concat in casts since only float is supported
            _wrap_concat_with_cast(ctx, node)
            return


@tf_op("Slice")
class Slice:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = Slice(T input, Index begin, Index size, @type Index)
        # T output = Slice(T data, @INTS axes, @INTS ends, @INTS starts)
        starts = node.inputs[1].get_tensor_value()
        sizes = node.inputs[2].get_tensor_value()
        ends = []
        for start, size in zip(starts, sizes):
            # get all elements
            if size == -1:
                dtype = ctx.get_dtype(node.input[1])
                utils.make_sure(dtype, "dtype of {} is None".format(node.input[1]))
                ends.append(np.iinfo(dtype).max)
            else:
                ends.append(start + size)
        ctx.remove_input(node, node.input[2])
        ctx.remove_input(node, node.input[1])
        node.set_attr("starts", starts)
        node.set_attr("ends", ends)


@tf_op("Gather")
class Gather:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        node.type = "Gather"


@tf_op("GatherV2")
class GatherV2:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # for GatherV2 axis come as input
        node.type = "Gather"
        axis = node.inputs[2].get_tensor_value()
        ctx.remove_input(node, node.input[2])
        node.set_attr("axis", axis)

INT64_MAX = np.iinfo(np.int64).max


def _make_gathernd_inner_loop(ctx, params, index, dtype):
    """create the inner loop for GatherNd."""
    # gather_cur = params
    # for (int i = 0; i < size(index); i++)
    #   gather_res = gather(gather_cur, index[i])
    scope_name = utils.make_name("gathernd_inner_loop")
    trip_node = ctx.make_node("Size", [index.output[0]])
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=np.bool))
    trip_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    cond_out_name = utils.make_name("cond_out")
    cur_name = utils.make_name("gather_cur")
    result_name = utils.make_name("res")

    # body graph creation
    g = ctx.create_new_graph_with_same_config()
    index_i = g.make_node("Gather", [index.output[0], trip_name], attr={"axis": 0})
    gather = g.make_node("Gather", [cur_name, index_i.output[0]], attr={"axis": 0})
    g.make_node("Squeeze", [gather.output[0]], attr={"axes": [0]}, outputs=[result_name])
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])

    g.add_graph_input(trip_name, TensorProto.INT64, [])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(cur_name, dtype, [])

    g.add_graph_output(cond_out_name, TensorProto.BOOL, [])
    g.add_graph_output(result_name, dtype, [])

    inner_loop = ctx.make_node("Loop", [trip_node.output[0],
                                        cond_const.output[0],
                                        params],
                               op_name_scope=scope_name, skip_conversion=False)
    inner_loop.set_body_graph_as_attr("body", g)
    return inner_loop


def make_gathernd(ctx, params, indices, output, scope_name, t_params, shapes, dtypes):
    """make GatherNd op."""
    # Tparams output = GatherNd(Tparams params, Tidx indices)
    scope_name = utils.make_name(scope_name)
    # reshape indices into [sum(indices[:-1]), indices[-1]]
    indices_shape = ctx.make_node("Shape", [indices], dtypes=[TensorProto.INT64])
    indices_size = ctx.make_node("Size", [indices])
    inner_shape = ctx.make_node("Slice",
                                [indices_shape.output[0]],
                                attr={"axes": [0], "ends": [INT64_MAX], "starts": [-1]},
                                dtypes=[TensorProto.INT64])
    outter_shape = ctx.make_node("Div",
                                 [indices_size.output[0], inner_shape.output[0]],
                                 dtypes=[TensorProto.INT64])
    flatten_shape = ctx.make_node("Concat",
                                  [outter_shape.output[0], inner_shape.output[0]],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    flatten_indices = ctx.make_node("Reshape", [indices, flatten_shape.output[0]])

    # outter loop for each index
    # for (int i=0; i<outter_shape; i++) inner_loop(params, flatten_indices[i])
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=np.bool))
    dummy_const = ctx.make_const(utils.make_name("dummy"), np.ones((), dtype=np.int64))

    # body graph creation
    g = ctx.create_new_graph_with_same_config()
    trip_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    cond_out_name = utils.make_name("cond_out")
    dummy_name = utils.make_name("dummy")
    dummy_out_name = utils.make_name("dummy_out")
    result_name = utils.make_name("res")

    index = g.make_node("Gather", [flatten_indices.output[0], trip_name], attr={"axis": 0})
    index_squeeze = g.make_node("Squeeze", [index.output[0]], attr={"axes": [0]})
    # inner loop to gather result
    inner_loop = _make_gathernd_inner_loop(g, params, index_squeeze, t_params)
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])
    g.make_node("Identity", [dummy_name], outputs=[dummy_out_name])
    g.make_node("Identity", [inner_loop.output[0]], outputs=[result_name])

    g.add_graph_input(trip_name, TensorProto.INT64, [])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(dummy_name, t_params, [])

    g.add_graph_output(cond_out_name, TensorProto.BOOL, [])
    g.add_graph_output(dummy_out_name, t_params, [])
    g.add_graph_output(result_name, t_params, [])

    gathernd_loop = ctx.make_node("Loop",
                                  [outter_shape.output[0], cond_const.output[0], params],
                                  output_count=2,
                                  op_name_scope=scope_name, skip_conversion=False)
    gathernd_loop.set_body_graph_as_attr("body", g)

    # reshape to target shape
    # output shape of gathernd: indices.shape[:-1] + gathernd_output.shape[1:]
    inner_loop_shape = ctx.make_node("Shape", [gathernd_loop.output[1]], dtypes=[TensorProto.INT64])
    # workaround in case gathernd_loop is 1-dimensional
    one_const = ctx.make_const(utils.make_name("one"), np.array([1], dtype=np.int64))
    inner_loop_shape_ = ctx.make_node("Concat",
                                      [inner_loop_shape.output[0], one_const.output[0]],
                                      attr={"axis": 0},
                                      dtypes=[TensorProto.INT64])
    output_inner_shape = ctx.make_node("Slice",
                                       [inner_loop_shape_.output[0]],
                                       attr={"axes": [0], "ends": [INT64_MAX], "starts": [1]},
                                       dtypes=[TensorProto.INT64])

    indices_outter_shape = ctx.make_node("Slice",
                                         [indices_shape.output[0]],
                                         attr={"axes": [0], "ends": [-1], "starts": [0]},
                                         dtypes=[TensorProto.INT64])
    output_shape_ = ctx.make_node("Concat",
                                  [indices_outter_shape.output[0], output_inner_shape.output[0]],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    output_shape = ctx.make_node("Slice",
                                 [output_shape_.output[0]],
                                 attr={"axes": [0], "ends": [-1], "starts": [0]},
                                 dtypes=[TensorProto.INT64])
    ctx.make_node("Reshape",
                  [gathernd_loop.output[1], output_shape.output[0]],
                  outputs=[output],
                  shapes=shapes,
                  dtypes=dtypes)


@tf_op("GatherNd")
class GatherNd:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # Tparams output = GatherNd(Tparams params, Tidx indices)
        params = node.input[0]
        indices = node.input[1]
        output = node.output[0]
        # same as the attr Tparams
        t_params = ctx.get_dtype(params)
        utils.make_sure(t_params, "Dtype of {} is None".format(indices))
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        make_gathernd(ctx, params, indices, output, node.name, t_params, shapes, dtypes)


@tf_op("Split")
class Split:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = Split(int32 split_dim, T value, @int num_split)
        # T outputs = Split(T input, @INT axis, @INTS split)
        split_dims = node.inputs[0].get_tensor_value()
        ctx.remove_input(node, node.input[0])
        node.set_attr("axis", split_dims)


@tf_op("SplitV")
class SplitV:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = SplitV(T value, Tlen size_splits, int32 split_dim, @int num_split, @type Tlen)
        # T outputs = Split(T input, @INT axis, @INTS split)
        node.type = "Split"
        split = node.inputs[1].get_tensor_value()
        split_dims = node.inputs[2].get_tensor_value()
        ctx.remove_input(node, node.input[2])
        ctx.remove_input(node, node.input[1])
        node.set_attr("split", split)
        node.set_attr("axis", split_dims)


@tf_op("ExpandDims")
class ExpandDims:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T output = ExpandDims(T input, Tdim dim, @type Tdim)
        # T reshaped = Reshape-1(T data, @ints consumed_inputs, @int64 shape)
        # T expanded = Unsqueeze-1(T data, @ints axes)
        shape = ctx.get_shape(node.output[0])
        if shape is not None and shape.count(-1) < 2:
            # tensorflow already infers the output shape so we can just take it
            shape = ctx.get_shape(node.output[0])
            node.type = "Reshape"
            ctx.remove_input(node, node.input[1])
            node.set_attr("shape", shape)
            return

        # if there is more than one -1 in the shape, Reshape won't support.
        dim_node = node.inputs[1]
        if dim_node.is_const():
            node.type = "Unsqueeze"
            dim = dim_node.get_tensor_value()
            if dim < 0:
                input_rank = len(ctx.get_shape(node.input[0]))
                dim = dim + input_rank + 1
            node.set_attr("axes", [dim])
            ctx.remove_input(node, node.input[1])
            return
        raise ValueError("non-const dim is not supported")

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T output = ExpandDims(T input, Tdim dim, @type Tdim), dim is 0-D scalar.
        # T reshaped = Reshape-5(T data, int64 shape)
        # T expanded = Unsqueeze-1(T data, @ints axes)
        # FIXME: this was in the OPSET5 table
        shape = ctx.get_shape(node.output[0])
        if shape is not None and shape.count(-1) < 2:
            # tensorflow already infers the output shape so we can just take it
            shape_name = utils.make_name(node.name)
            ctx.make_const(shape_name, np.array(shape, dtype=np.int64))
            node.type = "Reshape"
            node.input[1] = shape_name
            return

        # if there is more than one -1 in the shape, Reshape won't support.
        dim_node = node.inputs[1]
        if dim_node.is_const():
            node.type = "Unsqueeze"
            dim = dim_node.get_tensor_value()
            if dim < 0:
                input_rank = len(ctx.get_shape(node.input[0]))
                dim = dim + input_rank + 1
            node.set_attr("axes", [dim])
            ctx.remove_input(node, node.input[1])
            return
        raise ValueError("non-const dim is not supported")


@tf_op("StridedSlice")
class StridedSlice:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
        not_supported_attr = ["new_axis_mask"]
        for attr_name in not_supported_attr:
            attr = node.get_attr(attr_name)
            if attr is not None and attr.i != 0:
                raise ValueError("StridedSlice: attribute " + attr_name + " not supported")
        input_shape = ctx.get_shape(node.input[0])
        begin = node.inputs[1].get_tensor_value(as_list=False)
        end = node.inputs[2].get_tensor_value(as_list=False)
        strides = node.inputs[3].get_tensor_value(as_list=False)
        max_size = np.iinfo(begin.dtype).max
        end_mask = node.get_attr("end_mask")
        end_mask = end_mask.i if end_mask is not None else 0
        begin_mask = node.get_attr("begin_mask")
        begin_mask = begin_mask.i if begin_mask is not None else 0
        shrink_axis_mask = node.get_attr("shrink_axis_mask")
        shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
        ellipsis_mask = node.get_attr("ellipsis_mask")
        ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
        new_begin = []
        new_end = []
        axes = []
        # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
        needs_squeeze = []
        for idx, begin_item in enumerate(begin):
            end_item = end[idx]
            if strides[idx] != 1:
                raise ValueError("StridedSlice: only strides=1 is supported")
            axes.append(idx)

            if (ellipsis_mask >> idx) & 1:
                new_begin.append(0)
                new_end.append(max_size)
                continue

            # an implicit condition is stride == 1 (checked in above)
            if begin_item < 0 and end_item == 0:
                end_item = max_size

            mask = (shrink_axis_mask >> idx) & 1
            if mask != 0:
                new_begin.append(begin_item)
                end_item = begin_item + 1 if begin_item != -1 else max_size
                new_end.append(end_item)
                needs_squeeze.append(idx)
                continue

            mask = (begin_mask >> idx) & 1
            if mask != 0:
                new_begin.append(0)
            else:
                new_begin.append(begin_item)

            mask = (end_mask >> idx) & 1
            if mask != 0:
                new_end.append(max_size)
            else:
                new_end.append(end_item)

        node.set_attr("starts", new_begin)
        node.set_attr("ends", new_end)
        node.set_attr("axes", axes)
        node.type = "Slice"
        ctx.remove_input(node, node.input[3])
        ctx.remove_input(node, node.input[2])
        ctx.remove_input(node, node.input[1])
        nodes = [node]
        if needs_squeeze:
            name = utils.make_name(node.name)
            squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
            squeeze_node.set_attr("axes", needs_squeeze)
            nodes.append(squeeze_node)
            input_dtype = ctx.get_dtype(node.output[0])
            ctx.set_dtype(squeeze_node.output[0], input_dtype)
            ctx.copy_shape(node.output[0], squeeze_node.output[0])

        # onnx slice as of opset 7 does only take float tensors ... cast if needed
        input_dtype = ctx.get_dtype(node.input[0])
        if input_dtype != onnx_pb.TensorProto.FLOAT:
            if node.inputs[0].type == "Cast" and len(ctx.find_output_consumers(node.inputs[0].output[0])) == 1:
                # override the previous cast
                cast_node = node.inputs[0]
            else:
                cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
                nodes.insert(0, cast_node)
            cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
            ctx.copy_shape(node.input[0], cast_node.output[0])
            # undo the cast afer slice
            name = utils.make_name(node.name)
            cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
            cast_node.set_attr("to", input_dtype)
            ctx.set_dtype(cast_node.output[0], input_dtype)
            ctx.copy_shape(node.output[0], cast_node.output[0])
            nodes.append(cast_node)


@tf_op("Cast")
class Cast:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
        # T2 output = Cast(T1 input, @STRING to)
        dst = node.get_attr("to")
        dst = tf2onnx.utils.ONNX_DTYPE_NAMES[dst]
        node.set_attr("to", dst)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass


@tf_op("TopKV2")
class TopKV2:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # T values, int32 indices = TopKV2(T input, int32 k, @bool sorted=true, @realnumbertype T)
        # T values, I indices = TopK(T x, @int axis=-1, @int k). I: int64
        topk_node_name = node.name
        topk_output1 = node.output[0]
        topk_output2 = node.output[1]

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        k = node.inputs[1].get_tensor_value()
        ctx.remove_node(topk_node_name)
        new_topk_name = utils.make_name(topk_node_name)
        new_topk_node = ctx.make_node("TopK", [node.input[0]],
                                      outputs=[topk_output1, utils.port_name(new_topk_name, 1)],
                                      name=new_topk_name, attr={"k": k},
                                      shapes=shapes, dtypes=[dtypes[0], onnx_pb.TensorProto.INT64])

        new_cast_name = utils.make_name(topk_node_name)
        cast_to_int32 = ctx.make_node("Cast", [new_topk_node.output[1]], outputs=[topk_output2],
                                      name=new_cast_name, attr={"to": onnx_pb.TensorProto.INT32},
                                      shapes=[shapes[1]], dtypes=[onnx_pb.TensorProto.INT32])


@tf_op("Tile")
class Tile:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # onnx wants shape input to be int64
        _convert_shapenode_to_int64(ctx, node, 1)


@tf_op("Pack")
class Pack:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # hack to make up for the missing onnx pack op
        axis = node.get_attr("axis").i
        if axis < 0:
            axis += len(ctx.get_shape(node.input[0])) + 1

        inputs = []
        dtype = None
        # insert Unsqueeze on each input
        for i, n in enumerate(node.inputs):
            dtype = ctx.get_dtype(node.input[i])
            shape = ctx.get_shape(node.input[i])
            new_node = ctx.make_node("Unsqueeze", [node.input[i]], op_name_scope=node.name, attr={"axes": [axis]},
                                     shapes=[shape], dtypes=[dtype])
            output_name = new_node.output[0]
            node.input[i] = output_name
            inputs.append(output_name)

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        # concat all unqueezes
        concat = ctx.make_node("Concat", inputs, op_name_scope=node.name, attr={"axis": axis},
                               shapes=shapes, dtypes=dtypes)
        ctx.replace_all_inputs(ctx.get_nodes(), node.output[0], concat.output[0])


@tf_op("Unpack")
class Unpack:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # hack to make up for the missing onnx unpack op
        axis = node.get_attr("axis").i
        # split the tensor into n outputs
        node.type = "Split"
        # for each output we need to squeeze axis
        for n in node.output:
            op_name = utils.make_name(node.name)
            squeeze_node = ctx.insert_new_node_on_output("Squeeze", n, name=op_name, axes=[axis])
            ctx.copy_shape(n, squeeze_node.output[0])
            ctx.copy_dtype(n, squeeze_node.output[0])


@tf_op("OneHot")
class OneHot:
    @classmethod
    def version_5(cls, ctx, node, **kwargs):
        # until there is no onehot op in onnx, a workaround using gather from eye
        indices_name = node.input[0]
        indices_shape = ctx.get_shape(indices_name)
        if len(indices_shape) != 1:
            # TODO: this works for rank=1 but tensorflow supports more than this.
            # Same principle should work but we need to implemtn our own eye.
            raise ValueError("onehot op: only rank1 is supported")
        axis = node.get_attr("axis")
        # axis becomes axis for gather
        node.set_attr("axis", 0)
        depth = node.inputs[1].get_tensor_value()
        on_val = node.inputs[2].get_tensor_value(as_list=False)
        on = on_val.tolist()
        off = node.inputs[3].get_tensor_value()
        eye = np.eye(depth, dtype=on_val.dtype)
        if on != 0:
            eye[eye == 1] = on
            eye[eye == 0] = off
        else:
            eye[eye == 0] = off
            eye[eye == 1] = on

        const_name = utils.make_name(node.name)
        ctx.make_const(const_name, eye)
        # setup gather inputs
        del node.input[:]
        node.input.append(const_name)
        node.input.append(indices_name)
        node.type = "Gather"
        if axis.i == 0:
            # TODO: revisit for rank > 1
            name = utils.make_name(node.name)
            transpose_node = ctx.insert_new_node_on_output("Transpose", node.output[0], name)
            ctx.copy_shape(node.output[0], transpose_node.output[0])

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T output = OneHot(uint8/int32/int64 input, T depth, T on-value, T off-value, @int axis, @dtype)
        # tf requires that dtype is same as on-value's and off-value's dtype
        # in ONNX, op's schema is (input, depth, value, @int axis), meaning of "value" is [off-value, on-value]
        # onnxruntime only supports int64
        output_dtype = ctx.get_dtype(node.input[2])
        if ctx.is_target(constants.TARGET_RS6) \
            and output_dtype not in [onnx_pb.TensorProto.INT64, onnx_pb.TensorProto.INT32]:
            logger.warning("unsupported dtype in onnxruntime, onehot-9 can't be used directly")
            cls.version_5(ctx, node, **kwargs)
            return

        depth = node.input[1]
        depth = ctx.make_node("Unsqueeze", [depth], attr={"axes": [0]}).output[0]

        on_value = node.input[2]
        off_value = node.input[3]
        on_value = ctx.make_node("Unsqueeze", [on_value], attr={"axes": [0]}).output[0]
        off_value = ctx.make_node("Unsqueeze", [off_value], attr={"axes": [0]}).output[0]
        off_on_value = ctx.make_node("Concat", [off_value, on_value], attr={"axis": 0}).output[0]

        indices = node.input[0]
        if ctx.is_target(constants.TARGET_RS6) and ctx.get_dtype(indices) != onnx_pb.TensorProto.INT64:
            indices = ctx.make_node("Cast", [indices], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        node.input[0] = indices

        if ctx.is_target(constants.TARGET_RS6) and ctx.get_dtype(depth) != onnx_pb.TensorProto.INT64:
            depth = ctx.make_node("Cast", [depth], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        node.input[1] = depth

        if ctx.is_target(constants.TARGET_RS6) and output_dtype != onnx_pb.TensorProto.INT64:
            off_on_value = ctx.make_node("Cast", [off_on_value], attr={"to": onnx_pb.TensorProto.INT64}).output[0]
        node.input[2] = off_on_value

        del node.input[3]

        if ctx.is_target(constants.TARGET_RS6) and output_dtype != onnx_pb.TensorProto.INT64:
            new_node_name = utils.make_name("onehot_output")
            new_node = ctx.insert_new_node_on_output("Cast", node.output[0], new_node_name, to=output_dtype)
            ctx.set_dtype(new_node.output[0], output_dtype)
            ctx.set_shape(new_node.output[0], ctx.get_shape(node.output[0]))


@tf_op("Shape")
class Shape:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        # out_type output = Shape(T input, @int32|int64 out_type), out_type by default int32
        # int64 output = Shape(T input)
        dtype = ctx.get_dtype(node.output[0])
        if dtype == onnx_pb.TensorProto.INT64:
            return
        op_name = utils.make_name(node.name)
        output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name)
        output_cast.set_attr("to", dtype)
        ctx.set_dtype(output_cast.output[0], dtype)
        ctx.copy_shape(node.output[0], output_cast.output[0])


@tf_op("IsNan", onnx_op="IsNaN")
class IsNan:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        pass
