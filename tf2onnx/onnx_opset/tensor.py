# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tensor
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

import numpy as np
from onnx import numpy_helper
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto

import tf2onnx
from tf2onnx import constants, utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import nn, math

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement


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
            output_cast = ctx.insert_new_node_on_output("Cast", output_name, name=node.child_name())
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(output_name, output_cast.output[0])


@tf_op("Size")
class Size:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass


@tf_op("Flatten")
class Flatten:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # no change for us
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)


@tf_op("Dropout")
class Dropout:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        pass


@tf_op("Identity")
class Identity:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        if node.inputs[0].is_const():
            # should not remove the identity node if it is output of the graph
            if node.output[0] in ctx.outputs:
                return
            # if identity has a const as input, remove it
            input_name = node.input[0]
            output_name = node.output[0]
            ctx.replace_all_inputs(ctx.get_nodes(), output_name, input_name)
            ctx.remove_node(node.name)


@tf_op("IdentityN")
class IdentityN:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        ctx.remove_node(node.name)
        for input_name, output_name in zip(node.input, node.output):
            ctx.replace_all_inputs(ctx.get_nodes(), output_name, input_name)


@tf_op("Reshape")
class Reshape:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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
            output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=node.child_name())
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(node.output[0], output_cast.output[0])


@tf_op("Squeeze")
class Squeeze:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


@tf_op("Transpose")
class Transpose:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T y = Transpose(T x, Tperm perm, @type Tperm)
        # T transposed = Transpose(T data, @INTS perm)
        if len(node.input) > 1:
            perm = node.inputs[1]
            if perm.is_const():
                # perms is passed as const
                dims = perm.get_tensor_value()
                ctx.remove_input(node, node.input[1])
                node.set_attr("perm", dims)
            else:
                utils.make_sure(False, "perm can't be dynamic in ONNX")
        else:
            # graph rewrite moved perm to attribute
            pass


@tf_op("Concat")
class Concat:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


@tf_op("ConcatV2")
class ConcatV2:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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
            raise RuntimeError("all inputs of {} are empty".format(node.name))

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
    def version_1(cls, ctx, node, **kwargs):
        # T output = Slice(T input, Index begin, Index size)
        # T output = Slice(T input, Tind starts, Tind ends, Tind axes, Tind steps)
        # "ends" are exclusive, "axes" and "steps" are optional, their default val are [0, ...] and 1
        input_tensor = node.input[0]
        starts = node.input[1]
        size = node.input[2]
        # in tf, size can be -1 which means all elem are taken, so size can't be added starts directly.
        # the way to make sure size are not less than 0: set "sizes"'s elem to be int_max if elem val is -1
        size_dtype = ctx.get_dtype(size)
        size_np_dtype = utils.map_onnx_to_numpy_type(size_dtype)
        if ctx.get_node_by_output(size).is_const() and ctx.get_node_by_output(starts).is_const():
            starts = ctx.get_node_by_output(starts).get_tensor_value()
            sizes = ctx.get_node_by_output(size).get_tensor_value()
            ends = []
            for start, size in zip(starts, sizes):
                # get all elements
                if size == -1:
                    dtype = ctx.get_dtype(node.input[1])
                    utils.make_sure(dtype, "dtype of {} is None".format(node.input[1]))
                    utils.make_sure(dtype, "dtype of {} is None".format(node.input[1]))
                    ends.append(np.iinfo(dtype).max)
                else:
                    ends.append(start + size)

        else:
            neg_one_val = np.array([-1]).astype(size_np_dtype)
            neg_one = ctx.make_const(utils.make_name("const"), neg_one_val).output[0]

            int_max_val = np.array([utils.get_max_value(size_np_dtype)]).astype(size_np_dtype)
            int_max = ctx.make_const(utils.make_name("largest_int_val"), int_max_val).output[0]

            size_are_neg_one_flag = ctx.make_node("Equal", [neg_one, size]).output[0]
            size_are_neg_one_flag = ctx.make_node("Cast", [size_are_neg_one_flag], attr={"to": size_dtype}).output[0]
            value_to_add = ctx.make_node("Mul", [int_max, size_are_neg_one_flag]).output[0]
            size_processed = ctx.make_node("Add", [size, value_to_add]).output[0]
            ends = ctx.make_node("Add", [starts, size_processed]).output[0]

        ctx.remove_node(node.name)
        inputs_map = {"data": input_tensor, "starts": starts, "ends": ends}
        kwargs = {**inputs_map, "outputs": node.output}
        _ = GraphBuilder(ctx).make_slice(kwargs, name=node.name)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)


@tf_op("Gather")
class Gather:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.type = "Gather"

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)


@tf_op("GatherV2")
class GatherV2:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # for GatherV2 axis come as input
        node.type = "Gather"
        axis = node.inputs[2].get_tensor_value()
        ctx.remove_input(node, node.input[2])
        node.set_attr("axis", axis)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)


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
    g.add_graph_input(trip_name, TensorProto.INT64, [1])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(cur_name, dtype, [])
    g.parent_graph = ctx

    index_i = g.make_node("Gather", [index.output[0], trip_name], attr={"axis": 0})
    gather = g.make_node("Gather", [cur_name, index_i.output[0]], attr={"axis": 0})
    g.make_node("Squeeze", [gather.output[0]], attr={"axes": [0]}, outputs=[result_name])
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])

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
    attr = {"axes": [0], "ends": [sys.maxsize], "starts": [-1]}
    inputs_map = {"data": indices_shape.output[0], **attr}
    inner_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    outter_shape = ctx.make_node("Div",
                                 [indices_size.output[0], inner_shape],
                                 dtypes=[TensorProto.INT64])
    flatten_shape = ctx.make_node("Concat",
                                  [outter_shape.output[0], inner_shape],
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

    g.add_graph_input(trip_name, TensorProto.INT64, [1])
    g.add_graph_input(cond_name, TensorProto.BOOL, [])
    g.add_graph_input(dummy_name, t_params, [])
    g.parent_graph = ctx

    index = g.make_node("Gather", [flatten_indices.output[0], trip_name], attr={"axis": 0})
    index_squeeze = g.make_node("Squeeze", [index.output[0]], attr={"axes": [0]})
    # inner loop to gather result
    inner_loop = _make_gathernd_inner_loop(g, params, index_squeeze, t_params)
    g.make_node("Identity", [cond_name], outputs=[cond_out_name])
    g.make_node("Identity", [dummy_name], outputs=[dummy_out_name])
    g.make_node("Identity", [inner_loop.output[0]], outputs=[result_name])

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
    attr = {"axes": [0], "ends": [sys.maxsize], "starts": [1]}
    inputs_map = {"data": inner_loop_shape_.output[0], **attr}
    output_inner_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    attr = {"axes": [0], "ends": [-1], "starts": [0]}
    inputs_map = {"data": indices_shape.output[0], **attr}
    indices_outter_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    output_shape_ = ctx.make_node("Concat",
                                  [indices_outter_shape, output_inner_shape],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    attr = {"axes": [0], "ends": [-1], "starts": [0]}
    inputs_map = {"data": output_shape_.output[0], **attr}
    output_shape = GraphBuilder(ctx).make_slice(inputs_map, dtypes=[TensorProto.INT64])
    ctx.make_node("Reshape",
                  [gathernd_loop.output[1], output_shape],
                  outputs=[output],
                  shapes=shapes,
                  dtypes=dtypes)


@tf_op("GatherNd", onnx_op="GatherND")
class GatherND:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # indicies input
        input1 = node.input[1]
        target_dtype = TensorProto.INT64
        if ctx.get_dtype(input1) != TensorProto.INT64:
            inp_cast = ctx.insert_new_node_on_input(node, "Cast", input1, to=target_dtype)
            ctx.copy_shape(input1, inp_cast.output[0])
            ctx.set_dtype(inp_cast.output[0], target_dtype)


@tf_op("ScatterNd", onnx_op="ScatterND")
class ScatterND:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):

        # onnx requires pre-generated tensor for data
        np_val = np.array([0], dtype=np.int64)
        onnx_tensor = numpy_helper.from_array(np_val, node.child_name())
        const_of_shape = ctx.insert_new_node_on_input(node, "ConstantOfShape", node.input[2], value=onnx_tensor)

        # cast edge to INT64 if not already
        input0 = const_of_shape.input[0]
        if ctx.get_dtype(input0) != TensorProto.INT64:
            ctx.insert_new_node_on_input(const_of_shape, "Cast", input0, to=TensorProto.INT64)

        # cast edge to INT64 if not already
        input0 = node.input[0]
        if ctx.get_dtype(input0) != TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", input0, to=TensorProto.INT64)

        # reorder inputs to match onnx
        node.input = [node.input[2], node.input[0], node.input[1]]


@tf_op("Split")
class Split:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Split(int32 split_dim, T value, @int num_split)
        # T outputs = Split(T input, @INT axis, @INTS split)
        split_dims = node.inputs[0].get_tensor_value()
        ctx.remove_input(node, node.input[0])
        node.set_attr("axis", split_dims)

    @classmethod
    def version_2(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)


@tf_op("SplitV")
class SplitV:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = SplitV(T value, Tlen size_splits, int32 split_dim, @int num_split, @type Tlen)
        # T outputs = Split(T input, @INT axis, @INTS split)
        node.type = "Split"
        split = node.inputs[1].get_tensor_value()
        split_dims = node.inputs[2].get_tensor_value()
        ctx.remove_input(node, node.input[2])
        ctx.remove_input(node, node.input[1])
        node.set_attr("split", split)
        node.set_attr("axis", split_dims)

    @classmethod
    def version_2(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)


@tf_op("ExpandDims")
class ExpandDims:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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
        dim_node = node.inputs[1]
        if dim_node.is_const():
            node.type = "Unsqueeze"
            dim = dim_node.get_tensor_value()
            # TODO: isn't this always a list ?
            if isinstance(dim, list):
                dim = dim[0]
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
    def version_1(cls, ctx, node, **kwargs):
        # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
        not_supported_attr = ["new_axis_mask"]
        for attr_name in not_supported_attr:
            attr = node.get_attr(attr_name)
            if attr is not None and attr.i != 0:
                raise ValueError("StridedSlice: attribute " + attr_name + " not supported")

        onnx_dtype = ctx.get_dtype(node.input[1])
        np_dtype = utils.ONNX_TO_NUMPY_DTYPE[onnx_dtype]
        max_size = np.iinfo(np_dtype).max
        begin = node.inputs[1].get_tensor_value()
        end = node.inputs[2].get_tensor_value()
        strides = node.inputs[3].get_tensor_value()
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
        # ellipsis: one bit at most can be 1. An ellipsis implicitly creates as many range specifications as
        # necessary to fully specify the sliced range for every dimension.
        # For example for a 4-dimensional tensor foo the slice foo[2, ..., 5:8] implies foo[2, :, :, 5:8]
        # NOTE: we ignore those axes denoted by ellipsis using `axes` attribute
        ellipsis_gap = 0
        for idx, begin_item in enumerate(begin):
            if strides[idx] != 1:
                raise ValueError("StridedSlice: only strides=1 is supported")
            if (ellipsis_mask >> idx) & 1:
                input_shape = ctx.get_shape(node.input[0])
                utils.make_sure(
                    input_shape is not None,
                    "StridedSlice op {} requires the shape of input".format(node.name)
                )
                ellipsis_gap = len(input_shape) - len(begin)
                continue

            # ignore ellipsis axes
            axes.append(idx + ellipsis_gap)
            end_item = end[idx]

            # an implicit condition is stride == 1 (checked in above)
            if begin_item < 0 and end_item == 0:
                end_item = max_size

            mask = (shrink_axis_mask >> idx) & 1
            if mask != 0:
                new_begin.append(begin_item)
                end_item = begin_item + 1 if begin_item != -1 else max_size
                new_end.append(end_item)
                needs_squeeze.append(idx + ellipsis_gap)
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

        out_dtypes = [ctx.get_dtype(node.output[0])]
        out_shapes = [ctx.get_shape(node.output[0])]
        ctx.remove_node(node.name)

        attr = {"starts": new_begin, "ends": new_end, "axes": axes}
        inputs_map = {"data": node.input[0], **attr}
        kwargs = {**inputs_map, "outputs": node.output}
        node = GraphBuilder(ctx).make_slice(kwargs, name=node.name, dtypes=out_dtypes, shapes=out_shapes)
        node = ctx.get_node_by_output(node)
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
        if ctx.opset < 9:
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

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # T output = Slice(T input, Index begin, Index end, Index strides
        #                 @int begin_mask, @int end_mask, @int ellipsis_mask
        #                 @int shrink_axis_mask, @int new_axis_mask)
        # T output = Slice(T input, Tind starts, Tind ends, Tind axes, Tind steps)
        # "ends" are exclusive, "axes" and "steps" are optional, their default val are [0, ...] and 1
        input_x = node.inputs[0]
        begin = node.inputs[1]
        end = node.inputs[2]
        strides = node.inputs[3]
        new_axis_mask = node.get_attr("new_axis_mask")
        new_axis_mask = new_axis_mask.i if new_axis_mask is not None else 0

        if begin.is_const() and end.is_const() and strides.is_const() \
                and all(val == 1 for val in strides.get_tensor_value()) \
                and new_axis_mask == 0:
            cls.version_1(ctx, node, **kwargs)
            return

        onnx_dtype = ctx.get_dtype(node.input[1])
        np_dtype = utils.ONNX_TO_NUMPY_DTYPE[onnx_dtype]

        # NOTE: Max op only supports float32, deal with overflow when cast back to int32
        # enable it after Max supports int32 and int64
        # max_size = utils.get_max_value(np_dtype)
        # min_size = utils.get_min_value(np_dtype)
        max_size = 1e9
        min_size = -1e9

        end_mask = node.get_attr("end_mask")
        end_mask = end_mask.i if end_mask is not None else 0
        begin_mask = node.get_attr("begin_mask")
        begin_mask = begin_mask.i if begin_mask is not None else 0
        ellipsis_mask = node.get_attr("ellipsis_mask")
        ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
        shrink_axis_mask = node.get_attr("shrink_axis_mask")
        shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
        if new_axis_mask != 0:
            unqueeze_at = []
            for bit in range(32):
                if (new_axis_mask >> bit) & 1 == 1:
                    unqueeze_at.append(bit)
                    begin_mask |= 1 << bit
                    end_mask |= 1 << bit
            input_x = ctx.make_node("Unsqueeze", [input_x.output[0]], {"axes": unqueeze_at})

        param_shape = ctx.get_shape(node.input[1]) or \
                      ctx.get_shape(node.input[2]) or \
                      ctx.get_shape(node.input[3])
        utils.make_sure(
            param_shape is not None,
            "StridedSlice op {} requires the shape of begin/end/strides".format(node.name)
        )
        param_rank = param_shape[0]
        # use in onnx graph to mask begin
        new_begin_mask = [1] * param_rank
        # use in onnx graph to mask end
        new_end_mask = [min_size] * param_rank
        # for shrink mask, if shrink mask is 1, set stride to be max_size
        shrink_strided_mask = [min_size] * param_rank
        axes = []
        # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
        needs_squeeze = []
        ellipsis_gap = 0
        for idx in range(param_rank):
            if (ellipsis_mask >> idx) & 1:
                input_shape = ctx.get_shape(input_x.output[0])
                utils.make_sure(
                    input_shape is not None,
                    "StridedSlice op {} requires the shape of input".format(node.name)
                )
                ellipsis_gap = len(input_shape) - param_rank
                # handle the redundant param
                new_begin_mask[idx] = 0
                new_end_mask[idx] = max_size
                axes.append(idx)
                continue

            # ignore ellipsis axes
            axes.append(idx + ellipsis_gap)

            mask = (shrink_axis_mask >> idx) & 1
            if mask != 0:
                shrink_strided_mask[idx] = max_size
                new_end_mask[idx] = max_size
                needs_squeeze.append(idx + ellipsis_gap)
                continue

            mask = (begin_mask >> idx) & 1
            if mask != 0:
                new_begin_mask[idx] = 0

            mask = (end_mask >> idx) & 1
            if mask != 0:
                new_end_mask[idx] = max_size

        out_dtypes = [ctx.get_dtype(node.output[0])]
        out_shapes = [ctx.get_shape(node.output[0])]
        ctx.remove_node(node.name)

        # mask begin
        new_begin_mask = np.array(new_begin_mask, dtype=np_dtype)
        if not np.all(new_begin_mask == 1):
            if begin.is_const() and strides.is_const():
                new_begin_vals = np.copy(begin.get_tensor_value(as_list=False))
                strides_vals = strides.get_tensor_value(as_list=False)
                idx1 = np.where(new_begin_mask == 0)
                idx2 = np.where(strides_vals < 0)
                idx3 = np.intersect1d(idx1, idx2)
                new_begin_vals[idx3] = max_size
                begin = ctx.make_const(utils.make_name("begin_masked"), new_begin_vals)
            else:
                begin_mask_const = ctx.make_const(utils.make_name("begin_mask"), np.equal(new_begin_mask, 0))
                zero_const = ctx.make_const(utils.make_name("zero_const"), np.zeros(1, dtype=np_dtype))
                max_const = ctx.make_const(utils.make_name("max_const"), np.array(max_size, dtype=np_dtype))
                op1 = ctx.make_node("Less", [strides.output[0], zero_const.output[0]], op_name_scope=node.name)
                op2 = ctx.make_node("And", [op1.output[0], begin_mask_const.output[0]], op_name_scope=node.name)
                begin = ctx.make_node("Where", [op2.output[0], max_const.output[0], begin.output[0]],
                                      op_name_scope=node.name)

        # mask end
        new_end_mask = np.array(new_end_mask, dtype=np_dtype)
        end_output = end.output[0]
        if not np.all(new_end_mask == min_size):
            if end.is_const() and strides.is_const():
                new_end_mask = np.maximum(end.get_tensor_value(as_list=False), new_end_mask)
                idx = np.where(new_end_mask == max_size)
                sign = np.sign(strides.get_tensor_value(as_list=False))[idx]
                new_end_mask[idx] = new_end_mask[idx] * sign
                end = ctx.make_const(utils.make_name("end_masked"), new_end_mask)
                end_output = end.output[0]
            else:
                # Overlay new_end_mask with specified end values.
                # Adjust max_size to min_size if steps are < 0
                max_const = ctx.make_const(utils.make_name("max_const"), np.array(max_size, dtype=np_dtype))
                min_const = ctx.make_const(utils.make_name("min_const"), np.array(min_size, dtype=np_dtype))
                zero_const = ctx.make_const(utils.make_name("zero_const"), np.zeros(1, dtype=np_dtype))
                end_mask_const = ctx.make_const(utils.make_name("end_mask"), np.array(new_end_mask, dtype=np_dtype))
                outputname = utils.make_name("{}__newendmask".format(node.name))
                new_end_mask = math.make_min_or_max_op(ctx, "Max", [end.output[0], end_mask_const.output[0]],
                                                       [outputname])
                op1 = ctx.make_node("Less", [strides.output[0], zero_const.output[0]], op_name_scope=node.name)
                op2 = ctx.make_node("Equal", [new_end_mask.output[0], max_const.output[0]], op_name_scope=node.name)
                op3 = ctx.make_node("And", [op2.output[0], op1.output[0]], op_name_scope=node.name)
                final_end = ctx.make_node("Where", [op3.output[0], min_const.output[0],
                                                    new_end_mask.output[0]], op_name_scope=node.name)
                end_output = final_end.output[0]

        # mask strides for shrink
        shrink_strided_mask = np.array(shrink_strided_mask, dtype=np_dtype)
        strides_output = strides.output[0]
        if not np.all(shrink_strided_mask == min_size):
            if strides.is_const():
                strides = ctx.make_const(
                    utils.make_name("strides_masked"),
                    np.maximum(strides.get_tensor_value(as_list=False), shrink_strided_mask)
                )
                strides_output = strides.output[0]
            else:
                shrink_strided_mask_const = ctx.make_const(
                    utils.make_name("strides_mask"),
                    np.array(shrink_strided_mask, dtype=np_dtype)
                )
                strides_output = utils.make_name("{}__strides".format(node.name))
                math.make_min_or_max_op(
                    ctx, "Max",
                    [strides.output[0], shrink_strided_mask_const.output[0]],
                    [strides_output]
                )
        # create axes input
        axes_const = ctx.make_const(
            utils.make_name("slice_axes"),
            np.array(axes, dtype=np_dtype)
        )
        axes_output = axes_const.output[0]

        inputs_map = {
            "data": input_x.output[0],
            "starts": begin.output[0],
            "ends": end_output,
            "steps": strides_output,
            "axes": axes_output
        }
        kwargs = {**inputs_map, "outputs": node.output}
        node = GraphBuilder(ctx).make_slice(kwargs, name=node.name, dtypes=out_dtypes, shapes=out_shapes)
        node = ctx.get_node_by_output(node)
        if needs_squeeze:
            squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], node.child_name())
            squeeze_node.set_attr("axes", needs_squeeze)
            input_dtype = ctx.get_dtype(node.output[0])
            ctx.set_dtype(squeeze_node.output[0], input_dtype)
            ctx.copy_shape(node.output[0], squeeze_node.output[0])


@tf_op("Cast")
class Cast:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
        # T2 output = Cast(T1 input, @STRING to)
        dst = node.get_attr("to")
        dst = tf2onnx.utils.ONNX_DTYPE_NAMES[dst]
        node.set_attr("to", dst)

    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        pass


@tf_op("TopKV2", onnx_op="TopK")
class TopKV2:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # onnx only supports input K as a 1D tesor with dtype int64
        # while in tf, K is a 0D tensor with dtype int32
        dtypes = node.output_dtypes
        k_0d = node.input[1]
        cast = ctx.make_node("Cast", [k_0d], attr={"to": onnx_pb.TensorProto.INT64})
        k_1d = ctx.make_node("Unsqueeze", cast.output, attr={"axes": [0]})
        ctx.replace_input(node, k_0d, k_1d.output[0])
        # cast the index output to int32
        cast_out = ctx.insert_new_node_on_output("Cast", node.output[1], name=utils.make_name(node.name), to=dtypes[1])
        ctx.set_dtype(cast_out.output[0], dtypes[1])
        ctx.copy_shape(node.output[1], cast_out.output[0])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # opset 11 supports negative axis, and new attrs 'largest' and 'sorted'
        # the core logic doesn't change, using defaults for new attrs
        cls.version_10(ctx, node, **kwargs)


@tf_op("Tile")
class Tile:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # onnx wants shape input to be int64
        _convert_shapenode_to_int64(ctx, node, 1)


@tf_op("Pack")
class Pack:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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
    def version_1(cls, ctx, node, **kwargs):
        # hack to make up for the missing onnx unpack op
        # squeeze does not support negative axis
        axis = node.get_attr("axis").i
        if axis < 0:
            shape = ctx.get_shape(node.input[0])
            utils.make_sure(shape is not None, "shape of unpack input is None: {}".format(node.input[0]))
            axis += len(shape)
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
    def version_1(cls, ctx, node, **kwargs):
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
        # T output = OneHot(T input, T depth, T on-value, T off-value, @int axis, @dtype)
        # tf requires that dtype is same as on-value's and off-value's dtype
        # in ONNX, op's schema is (input, depth, value, @int axis), meaning of "value" is [off-value, on-value]
        output_dtype = ctx.get_dtype(node.input[2])

        depth = node.input[1]
        depth = ctx.make_node("Unsqueeze", [depth], attr={"axes": [0]}).output[0]

        on_value = node.input[2]
        off_value = node.input[3]
        on_value = ctx.make_node("Unsqueeze", [on_value], attr={"axes": [0]}).output[0]
        off_value = ctx.make_node("Unsqueeze", [off_value], attr={"axes": [0]}).output[0]
        off_on_value = ctx.make_node("Concat", [off_value, on_value], attr={"axis": 0}).output[0]

        indices = node.input[0]
        node.input[2] = off_on_value

        del node.input[3]

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_9(ctx, node, **kwargs)


@tf_op("Shape")
class Shape:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
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


@tf_op("BatchToSpaceND", onnx_op="DepthToSpace")
class BatchToSpace:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # block_shape impacts Transpose 'perm' attribute values.
        # must be available at compile time
        utils.make_sure(node.inputs[1].is_const(), 'only support constant block_shape value.')

        block_shape = node.inputs[1].get_tensor_value(False)
        blocklen = len(block_shape)
        xlen = len(ctx.get_shape(node.input[0]))

        # if 3d or 4d tensor & square 2d block_shape , can optimize
        cond1 = xlen in [3, 4]
        cond2 = node.inputs[2].is_const()
        cond3 = blocklen == 2 and block_shape[0] == block_shape[1]
        if cond1 and cond2 and cond3:
            # https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d.html
            # the above link says the data format of input tensor should be (batch, spatial_shape, remaining_shape)
            # and we only support 3D and 4D here, and the data format is NHC and NHWC
            # onnx op "DepthToSpace" does the same work on input tensor except that it works on "C",
            # and it only supports NCHW
            # T out = BatchToSpaceND(T input, int32 block_shape, int32 crops)
            input_tensor = node.inputs[0]
            input_shape = ctx.get_shape(input_tensor.output[0])
            crops = node.inputs[2].get_tensor_value()

            # NHWC TO CNHW, so onnx op will work on "N" which is the same as tensorflow
            if len(input_shape) == 3:
                # insert automatically an Unsqueeze op if the input is 3d
                unsqz1 = ctx.make_node("Unsqueeze", input_tensor.output, {"axes": [3]})
                trans1 = ctx.make_node("Transpose", unsqz1.output, {"perm": [3, 0, 1, 2]})
            else:
                trans1 = ctx.make_node("Transpose", input_tensor.output, {"perm": [3, 0, 1, 2]})
            reorganize_node = ctx.make_node(node.type, trans1.output, attr={"blocksize": block_shape[0]})
            trans2 = ctx.make_node("Transpose", reorganize_node.output, {"perm": [1, 2, 3, 0]})

            # implement crop logic, the data format is NHWC
            slice_axis = [1, 2]
            top, bottom = crops[0]
            left, right = crops[1]
            starts = [top, left]
            ends = []
            for end in [bottom, right]:
                if end != 0:
                    ends.append(-end)
                else:
                    ends.append(np.iinfo(np.int32).max)

            attr = {"axes": slice_axis, "ends": ends, "starts": starts}
            inputs_map = {"data": trans2.output[0], **attr}
            dtypes = node.output_dtypes
            shapes = node.output_shapes

            if len(input_shape) == 3:
                # add a squeeze op to convert output into 3d
                kwargs = {**inputs_map}
                ctx.remove_node(node.name)
                slice1 = GraphBuilder(ctx).make_slice(kwargs)
                ctx.make_node("Squeeze", [slice1], {"axes": [3]},
                              outputs=node.output, name=node.name, dtypes=dtypes, shapes=shapes)
            else:
                kwargs = {**inputs_map, "outputs": node.output}
                ctx.remove_node(node.name)
                GraphBuilder(ctx).make_slice(kwargs, name=node.name, dtypes=dtypes, shapes=shapes)
        else:
            def mknode(optype, inputs, attrs=None):
                nodename = utils.make_name(node.name + '_' + optype.lower())
                return ctx.make_node(optype, inputs, attrs, name=nodename)

            def mkconst(desc, val, dtype=np.int64):
                nodename = utils.make_name(node.name + '_' + desc)
                const_node = ctx.make_const(utils.make_name(nodename), val.astype(dtype))
                return const_node.output[0]

            # support non 3D/4D tensors and dynamic crop vals
            # dynamic slice starts at opset 10
            utils.make_sure(ctx.opset >= 11, 'non-4D tensor or non-const crops require opset 11')

            input0 = node.input[0]
            input2 = node.input[2]

            # const vals
            int_max_const = mkconst('int_max', np.array([utils.get_max_value(np.int64)]))
            one_const = mkconst('_const_one', np.array([1]))
            minus1_const = mkconst('_const_minus1', np.array([-1]))
            blocklen_resize_const = mkconst('_const_blocklen_resize', np.array([-1, blocklen]))
            blocklenplus1_const = mkconst('_const_blocklenplus1', np.array([blocklen + 1]))
            block_shape_const = mkconst('_const_block_shape', block_shape)

            x_shape = ctx.insert_new_node_on_input(node, 'Shape', node.input[0])

            # get the spatial and depth (i.e remaining) dimensions
            # compute target spatial dimensions by multiplying block_shape
            spatial = mknode('Slice', [x_shape.output[0], one_const, blocklenplus1_const])
            depth = mknode('Slice', [x_shape.output[0], blocklenplus1_const, int_max_const])
            target_spatial = mknode('Mul', [spatial.output[0], block_shape_const])

            # shape to use before shuffle  (part 1)
            ccat1 = mknode('Concat', [spatial.output[0], block_shape_const], {'axis': 0})
            re1 = mknode('Reshape', [ccat1.output[0], blocklen_resize_const])
            tr1 = mknode('Transpose', [re1.output[0]])
            interleave = mknode('Reshape', [tr1.output[0], minus1_const])
            shape1 = mknode('Concat', [minus1_const, interleave.output[0], depth.output[0]], {'axis': 0})

            # shape to use before shuffle (part 2)
            g1 = list(range(2, 2 * blocklen + 1, 2))
            g2 = list(range(1, 2 * blocklen + 1, 2))
            g = g1 + [0] + g2 + list(range(0, xlen + blocklen)[1 + 2 * blocklen:])

            # permutation values for shuffling
            p = np.asarray(range(0, xlen + blocklen))
            p[0] = blocklen
            p[1] = blocklen + 1
            p[2] = 0
            for i in range(3, blocklen * 2 + 1):
                p[i] = p[i - 2] + 1

            # reshape to create moving blocks, shuffle, and reshape to target_spatial
            indices = mkconst('_indicies_const', np.asarray(g))
            gather = mknode('Gather', [shape1.output[0], indices])
            x2 = mknode('Reshape', [input0, gather.output[0]])
            tr2 = mknode('Transpose', [x2.output[0]], {'perm': np.array(p)})
            shape2 = mknode('Concat', [minus1_const, target_spatial.output[0], depth.output[0]], {'axis': 0})
            x3 = mknode('Reshape', [tr2.output[0], shape2.output[0]])

            # crop axes
            slice_starts_const1 = mkconst('_slicestart1_const', np.asarray([0, 0]))
            slice_starts_const2 = mkconst('_slicestart2_const', np.asarray([1, utils.get_max_value(np.int64)]))
            slice_ends_const1 = mkconst('_sliceend1_const', np.asarray([1, 0]))
            slice_ends_const2 = mkconst('_sliceend2_const', np.asarray([2, utils.get_max_value(np.int64)]))
            axes_const = mkconst('_sliceaxes_const', np.asarray(range(1, blocklen + 1)))
            crop = mknode('Cast', [input2], {'to': TensorProto.INT64})
            crop_transposed = mknode('Transpose', [crop.output[0]])
            crop_starts = mknode('Slice', [crop_transposed.output[0], slice_starts_const1, slice_starts_const2])
            crop_ends = mknode('Slice', [crop_transposed.output[0], slice_ends_const1, slice_ends_const2])
            crop_starts_squeeze = mknode('Squeeze', [crop_starts.output[0]], {'axes': [0]})
            crop_ends_squeeze = mknode('Squeeze', [crop_ends.output[0]], {'axes': [0]})
            end_range = mknode('Sub', [target_spatial.output[0], crop_ends_squeeze.output[0]])
            orig_shape = node.output_shapes
            orig_dtypes = node.output_dtypes
            ctx.remove_node(node.name)
            ctx.make_node('Slice', [x3.output[0], crop_starts_squeeze.output[0], end_range.output[0], axes_const],
                          name=node.name, outputs=node.output, shapes=orig_shape, dtypes=orig_dtypes)


@tf_op("SpaceToBatchND", onnx_op="SpaceToDepth")
class SpaceToBatch:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # block_shape impacts Transpose 'perm' attribute values.
        # must be available at compile time
        utils.make_sure(node.inputs[1].is_const(), 'only support constant block_shape value.')

        block_shape = node.inputs[1].get_tensor_value(False)
        blocklen = len(block_shape)
        xlen = len(ctx.get_shape(node.input[0]))

        # if 3d or 4d tensor & square 2d block_shape , can optimize
        cond1 = xlen in [3, 4]
        cond2 = node.inputs[2].is_const()
        cond3 = blocklen == 2 and block_shape[0] == block_shape[1]
        if cond1 and cond2 and cond3:
            # https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
            # the above link says the data format of input tensor should be (batch, spatial_shape, remaining_shape)
            # and we only support 4D here, so the data format is NHWC
            # onnx op "SpaceToDepth" does the same work on input tensor except that it works on "C",
            # and it only supports NCHW
            # T out = SpaceToBatchND(T input, int32 block_shape, int32 crops)
            input_tensor = node.inputs[0]
            shapes = [ctx.get_shape(node.output[0])]
            dtypes = [ctx.get_dtype(node.output[0])]

            # implement pads logic, the data format is NHWC
            paddings = node.inputs[2].get_tensor_value()
            top, bottom = paddings[0]
            left, right = paddings[1]
            pads = [0, top, left, 0,
                    0, bottom, right, 0]
            ctx.remove_node(node.name)
            if ctx.opset <= 10:
                pad_op = ctx.make_node("Pad", input_tensor.output, attr={"pads": pads})
            else:
                # TODO: we should be able to support dynamic input here.
                pads_name = utils.make_name(node.name)
                ctx.make_const(name=pads_name, np_val=np.array(pads, dtype=np.int64))
                pad_op = ctx.make_node("Pad", [input_tensor.output[0], pads_name])

            # NHWC TO CNHW, so onnx op will work on "N" which is the same as tensorflow
            trans1 = ctx.make_node("Transpose", pad_op.output, {"perm": [3, 0, 1, 2]})
            reorganize_node = ctx.make_node(node.type, trans1.output, attr={"blocksize": block_shape[0]})
            ctx.make_node("Transpose", reorganize_node.output, {"perm": [1, 2, 3, 0]},
                          name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
        else:
            def mknode(optype, inputs, attrs=None):
                nodename = utils.make_name(node.name + '_' + optype.lower())
                return ctx.make_node(optype, inputs, attrs, name=nodename)

            def mkconst(desc, val, dtype=np.int64):
                nodename = utils.make_name(node.name + '_' + desc)
                const_node = ctx.make_const(utils.make_name(nodename), val.astype(dtype))
                return const_node.output[0]

            # support non 3D/4D tensors and dynamic pad vals
            # dynamic slice starts at opset 10
            utils.make_sure(ctx.opset >= 11, 'non-4D tensor or non-const pads require opset 11')

            input0 = node.input[0]
            input2 = node.input[2]

            # const vals
            int_max_const = mkconst('int_max', np.array([utils.get_max_value(np.int64)]))
            zero_const = mkconst('_zero_const', np.array([0]))
            one_const = mkconst('_one_const', np.array([1]))
            minus1_const = mkconst('_minus1_const', np.array([-1]))
            blocklen_resize_const = mkconst('_blocklen_resize_const', np.array([-1, blocklen]))
            blocklenplus1_const = mkconst('_blocklenplus1_const', np.array([blocklen + 1]))
            filltop_const = mkconst('_filltop_const', np.array([1, 0, 0, 0]))
            fillbottom_const = mkconst('_bottom_const', np.array([0, 0, 1, 0]))
            block_shape_const = mkconst('_block_shape_const', block_shape)

            x_shape = ctx.insert_new_node_on_input(node, 'Shape', node.input[0])
            x_rank = mknode('Size', [x_shape.output[0]])

            # pad x prior to compute
            pad = mknode('Cast', [input2], {'to': TensorProto.INT64})
            pad_shape = mknode('Shape', [pad.output[0]])
            pad_rank = mknode('Slice', [pad_shape.output[0], zero_const, one_const])
            pad_gap = mknode('Sub', [x_rank.output[0], pad_rank.output[0]])
            gapminus1 = mknode('Sub', [pad_gap.output[0], one_const])
            gapminus1fillbot = mknode('Mul', [fillbottom_const, gapminus1.output[0]])
            padfilltop = mknode('Pad', [pad.output[0], filltop_const])
            padfilltopbottom = mknode('Pad', [padfilltop.output[0], gapminus1fillbot.output[0]])
            pad_t = mknode('Transpose', [padfilltopbottom.output[0]])
            pad1d = mknode('Reshape', [pad_t.output[0], minus1_const])

            # get the spatial and depth (i.e remaining) dimensions
            # compute reduced spatial dimensions by dividing block_shape
            x1 = mknode('Pad', [input0, pad1d.output[0]])
            x1_shape = mknode('Shape', [x1.output[0]])
            spatial = mknode('Slice', [x1_shape.output[0], one_const, blocklenplus1_const])
            depth = mknode('Slice', [x1_shape.output[0], blocklenplus1_const, int_max_const])
            reduced = mknode('Div', [spatial.output[0], block_shape_const])

            # reshape x into smaller blocks before shuffle
            ccat1 = mknode('Concat', [reduced.output[0], block_shape_const], {'axis': 0})
            reshape1 = mknode('Reshape', [ccat1.output[0], blocklen_resize_const])
            tr1 = mknode('Transpose', [reshape1.output[0]])
            interleave = mknode('Reshape', [tr1.output[0], minus1_const])
            shape1 = mknode('Concat', [minus1_const, interleave.output[0], depth.output[0]], {'axis': 0})
            x2 = mknode('Reshape', [x1.output[0], shape1.output[0]])

            # permutation values for shuffling
            p1 = list(range(2, 2 * blocklen + 1, 2))
            p2 = list(range(1, 2 * blocklen + 1, 2))
            perm = p1 + [0] + p2 + list(range(0, xlen + blocklen)[1 + 2 * blocklen:])

            tr2 = mknode('Transpose', [x2.output[0]], {'perm': perm})
            shape2 = mknode('Concat', [minus1_const, reduced.output[0], depth.output[0]], {'axis': 0})
            orig_shape = node.output_shapes
            orig_dtypes = node.output_dtypes
            ctx.remove_node(node.name)
            ctx.make_node('Reshape', [tr2.output[0], shape2.output[0]],
                          name=node.name, outputs=node.output, shapes=orig_shape,
                          dtypes=orig_dtypes)


@tf_op("IsInf", onnx_op="IsInf")
class IsInf:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        node_dtype = ctx.get_dtype(node.input[0])
        utils.make_sure(node_dtype, "Dtype of {} is None".format(node.name))
        if node_dtype not in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.DOUBLE]:
            raise ValueError("dtype " + str(node_dtype) + " is not supported in onnx for now")


@tf_op(["NonMaxSuppressionV2", "NonMaxSuppressionV3"], onnx_op="NonMaxSuppression")
class NonMaxSuppression:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # int32 = NonMaxSuppressionV2(T boxes, T scores, int32 max_output_size, T iou_threshold, T score_threshold)
        # int64 = NonMaxSuppression(T boxes, T scores, int64 max_output_size, T iou_threshold, T score_threshold),
        # T means float32 here, the last 3 params are optional
        # tf boxes is 2D ([boxes_num, 4]) while onnx is 3D ([num_batches, boxes_num, 4])
        # tf scores is 1D ([boxes_num])while onnx is 2D ([num_batches, num_classes, boxes_num])
        # onnx output is [num_selected_boxes, 3], the meaning of last dim is [batch_index, class_index, box_index]
        # while tf's output is [num_selected_boxes]
        ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[0], axes=[0])
        ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[1], axes=[0, 1])
        ctx.insert_new_node_on_input(node, "Cast", node.input[2], to=onnx_pb.TensorProto.INT64)
        # replace original node with nonmaxsurppress + slice + squeeze + cast
        dtypes = [ctx.get_dtype(node.output[0])]
        shapes = [ctx.get_shape(node.output[0])]
        ctx.remove_node(node.name)
        new_nonmaxsurppress = ctx.make_node(node.type, node.input).output[0]
        slice_op = GraphBuilder(ctx).make_slice({"data": new_nonmaxsurppress,
                                                 "axes": [1], "ends": [3], "starts": [2]})
        squeeze_op = ctx.make_node("Squeeze", [slice_op], attr={"axes": [1]})
        ctx.make_node("Cast", inputs=squeeze_op.output, attr={"to": onnx_pb.TensorProto.INT32},
                      name=node.name, outputs=node.output, dtypes=dtypes, shapes=shapes)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_10(ctx, node, **kwargs)


@tf_op("ReverseSequence")
class ReverseSequence:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # T output = Scan(int64 sequence_lens, variadic initial_state_and_scan_inputs, @graph body,
        #                 @ints directions,@int num_scan_inputs)
        seq_dim = node.get_attr("seq_dim")
        batch_dim = node.get_attr("batch_dim")
        batch_major = seq_dim.i == 1 and (batch_dim or batch_dim.i == 0)
        time_major = batch_dim.i == 1 and (seq_dim or seq_dim.i == 0)
        perm_val = None

        if not batch_major and not time_major:
            error_msg = "unsupported attributes, seq_dim:{}, batch_dim:{}".format(seq_dim, batch_dim)
            raise ValueError(error_msg)

        if time_major:
            old_shape = ctx.get_shape(node.input[0])
            old_dtype = ctx.get_dtype(node.input[0])
            perm_val = [1, 0]
            rank = len(old_shape)
            utils.make_sure(rank >= 2, "rank of reverse_sequence input {} is at least 2".format(node.input[0]))
            perm_val += list(range(2, rank))
            trans_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[0], perm=perm_val)
            new_shape = nn.spatial_map(old_shape, perm_val)
            ctx.set_shape(trans_node.output[0], new_shape)
            ctx.set_dtype(trans_node.output[0], old_dtype)

        # handle batch_major input
        node.type = "Scan"
        node.set_attr("num_scan_inputs", 1)
        input_dtype = ctx.get_dtype(node.input[0])
        input_shape = ctx.get_shape(node.input[0])

        g = ctx.create_new_graph_with_same_config()
        g.parent_graph = ctx
        g.add_graph_input('X', input_dtype, input_shape[2:])
        g.make_node('Identity', ['X'], outputs=['Y'])
        g.add_graph_output('Y', input_dtype, input_shape[2:])

        node.set_body_graph_as_attr("body", g)
        node.set_attr("directions", [1])  # reverse the scan input

        seq_len_dtype = ctx.get_dtype(node.input[1])
        if seq_len_dtype != onnx_pb.TensorProto.INT64:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[1])
            cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
            ctx.copy_shape(node.input[1], cast_node.output[0])

        if time_major:
            # get back to time_major
            op_name = utils.make_name(node.name)
            trans_back_node = ctx.insert_new_node_on_output("Transpose", node.output[0],
                                                            name=op_name, perm=perm_val)
            ctx.copy_dtype(node.output[0], trans_back_node.output[0])

        tmp = node.input[0]
        node.input[0] = node.input[1]
        node.input[1] = tmp

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # we cannot easily construct reverse_sequence equivalence in opset 9, so we will not support it
        # here. Actually using loops to do that is kind of meaningless since there will be performance
        # issue there for sure.
        raise NotImplementedError("ReverseSequence is not supported to convert in OPSET 9,"
                                  " if possible please try using OPSET 8, or OPSET >=10 instead.")

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # T output = ReverseSequence(T input, int64 sequence_lens, @int time_axis, @int batch_axis)
        seq_dim = node.get_attr("seq_dim")
        utils.make_sure(seq_dim is not None, "sequence dim must be given in {}".format(node.name))
        seq_dim = seq_dim.i
        batch_dim = node.get_attr_value("batch_dim", 0)

        ctx.remove_node(node.name)
        node = ctx.make_node(
            "ReverseSequence",
            node.input,
            outputs=node.output,
            attr={"batch_axis": batch_dim, "time_axis": seq_dim})

        seq_len_dtype = ctx.get_dtype(node.input[1])
        utils.make_sure(seq_len_dtype is not None, "dtype of {} is None".format(node.input[1]))
        target_dtype = TensorProto.INT64
        if seq_len_dtype != target_dtype:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=target_dtype)


@tf_op("ReverseV2")
class ReverseV2:
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # T output = ReverseV2(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # Implement tensorflow ReverseV2 op using multiple ReverseSequence (for each axis)
        # and Transpose ops. We sort the axis vector (if non-empty) at the start. Each axis can
        # be reversed only once (in tf) and so we can compute the transpose for each axis
        # (other than 0), feed the tensor to a ReverseSequence node and finally transpose again
        # to get back the original shape.

        axes_node = node.inputs[1]
        axes = axes_node.get_tensor_value(as_list=False)
        # Current support is for when axis is a 1D tensor.
        utils.make_sure(len(axes.shape) == 1 \
                        , "Currently no support for reverseV2 tensor axis")

        axes = axes.tolist()
        len_axes = len(axes)

        # Store input and output parameters of the ReverseV2 node.
        rv2_in_names = [node.input[0]]

        input_shape = ctx.get_shape(node.input[0])
        # Make sure input shape is not None
        utils.make_sure(input_shape is not None, "shape of {} is None".format(node.input[0]))

        input_rank = len(input_shape)

        rv2_node_name = node.name
        # ReverseV2 has a single output.
        rv2_output_dtypes = node.output_dtypes
        rv2_output_shapes = node.output_shapes

        const_name_root = rv2_node_name + '_Const'

        # Remove ReverseV2 node from graph.
        ctx.remove_node(rv2_node_name)

        # Variable to store input names for the next node.
        inputs = rv2_in_names

        new_node = None

        # Empty axis vector.
        if len_axes == 0:
            # Replace ReverseV2 with an identity block.
            new_node = ctx.make_node(
                "Identity",
                inputs=inputs,
                outputs=node.output,
                shapes=rv2_output_shapes,
                dtypes=rv2_output_dtypes,
                op_name_scope=rv2_node_name,
            )

        else:
            # For negative indices use the positive counterpart.
            for i, ax in enumerate(axes):
                if ax < 0:
                    axes[i] += input_rank

            axes = sorted(axes)

            orig_perm = list(range(input_rank))
            curr_perm = []

            # Add ReverseSequence nodes for each element of axis.
            for i in range(len_axes):

                axis = axes[i]

                curr_perm = orig_perm.copy()
                # Permutation indices relative to original tensor.
                curr_perm[axis], curr_perm[0] = curr_perm[0], curr_perm[axis]

                # Add a Transpose node if the axis != 0 (finish first due to sort).
                if axis != 0:
                    # Permutation indices for the transpose node relative to IN tensor shape.
                    new_node = ctx.make_node(
                        "Transpose",
                        inputs=inputs,
                        op_name_scope=rv2_node_name,
                        dtypes=rv2_output_dtypes,
                        attr={"perm": curr_perm}
                    )

                    inputs = [new_node.output[0]]

                # Add a Constant node (seq_len) for ReverseSequence.
                if ctx.opset >= 11:
                    batch_shape = ctx.make_node("Shape", [inputs[-1]])
                    const_one = ctx.make_const(utils.make_name(node.name + "_const_one"), np.array([1], dtype=np.int64))
                    const_two = ctx.make_const(utils.make_name(node.name + "_const_two"), np.array([2], dtype=np.int64))
                    batch_size = ctx.make_node("Slice",
                                               [batch_shape.output[0], const_one.output[0], const_two.output[0]])
                    input_shape = ctx.make_node("Shape", [node.input[0]])
                    const_axis = ctx.make_const(utils.make_name(node.name + "_const_axis"),
                                                np.array([axis], dtype=np.int64))
                    const_axis_next = ctx.make_const(utils.make_name(node.name + "_const_axis_next"),
                                                     np.array([axis + 1], dtype=np.int64))
                    input_axis = ctx.make_node("Slice",
                                               [input_shape.output[0], const_axis.output[0], const_axis_next.output[0]])
                    seq_array = ctx.make_node("Expand", [input_axis.output[0], batch_size.output[0]])
                    inputs.append(seq_array.output[0])
                else:
                    # Index 1 for the shape should not return 0
                    # since the input must have rank >= 2.
                    rs_batch_size = ctx.get_shape(inputs[-1])[1]
                    # Make sure rs_batch_size and input_shape[axis] are not -1 each
                    utils.make_sure(input_shape[axis] is not -1 \
                                    , "shape of axis {} is unknown".format(axis))
                    utils.make_sure(rs_batch_size is not -1 \
                                    , "ReverseSequence batch size for axis {} is unknown".format(axis))
                    seq_list = [input_shape[axis]] * rs_batch_size
                    seq_array = np.asarray(seq_list, dtype=np.int64)  # dtype should be int64
                    const_seq_name = utils.make_name(const_name_root)
                    new_node = ctx.make_const(name=const_seq_name, np_val=seq_array)
                    inputs.append(new_node.output[0])

                # Add a ReverseSequence node.

                # If processing for the final axis and the tensor shape permutation is
                # original then the output is fed to the output of the ReverseV2 node.
                #
                # Else a new output is created which is fed to a Transpose node.
                rs_out_name = node.output if \
                    ((i == len_axes - 1) and (curr_perm == orig_perm)) \
                    else None

                rs_out_shapes = None if rs_out_name is None else rv2_output_shapes

                new_node = ctx.make_node(
                    "ReverseSequence",
                    inputs=inputs,
                    op_name_scope=rv2_node_name,
                    outputs=rs_out_name,
                    shapes=rs_out_shapes,
                    dtypes=rv2_output_dtypes,
                    attr={"batch_axis": 1, "time_axis": 0}
                )

                inputs = [new_node.output[0]]

            # Additional transpose block is required if the current
            # permutation list is not the original one.
            if curr_perm != orig_perm:

                # Compute the required permutation list.
                if len_axes != 1:
                    for i, ax in enumerate(axes[::-1][1:]):
                        curr_perm[0], curr_perm[ax] = \
                            curr_perm[ax], curr_perm[0]

                # Add a Transpose node to restore shape.
                new_node = ctx.make_node(
                    "Transpose",
                    inputs=inputs,
                    op_name_scope=rv2_node_name,
                    outputs=node.output,
                    shapes=rv2_output_shapes,
                    dtypes=rv2_output_dtypes,
                    attr={"perm": curr_perm}
                )


@tf_op("Unique", onnx_op="Unique")
class Unique:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # opset 11 supports explicitly
        dtypes = node.output_dtypes
        if len(node.output) > 1:
            # cast to int64 if needed
            if dtypes[1] != onnx_pb.TensorProto.UINT64:
                cast_node = ctx.insert_new_node_on_output("Cast", node.output[1],
                                                          name=utils.make_name(node.name) + "_cast")
                cast_node.set_attr("to", dtypes[1])
                ctx.set_dtype(cast_node.output[0], dtypes[1])
                ctx.copy_shape(node.output[1], cast_node.output[0])
            # FIXME: the indices in onnx are not the same as in tensorflow.


@tf_op("MatrixDiagPart")
class MatrixDiagPart:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # MatrixDiagPart by slice and gather
        const_zero = ctx.make_const(utils.make_name(node.name) + 'const_zero', np.array([0]).astype(np.int64))
        const_zero_zero = ctx.make_const(utils.make_name(node.name) + 'const_zero_zero',
                                         np.array([0, 0]).astype(np.int64))
        const_one = ctx.make_const(utils.make_name(node.name) + 'const_one', np.array([1]).astype(np.int64))
        const_two = ctx.make_const(utils.make_name(node.name) + 'const_two', np.array([2]).astype(np.int64))
        const_negative_one = ctx.make_const(utils.make_name(node.name) + 'const_negative_one',
                                            np.array([-1]).astype(np.int64))
        const_negative_two = ctx.make_const(utils.make_name(node.name) + 'const_negative_two',
                                            np.array([-2]).astype(np.int64))
        const_negative_two_one = ctx.make_const(utils.make_name(node.name) + 'const_negative_two_one',
                                                np.array([-2, -1]).astype(np.int64))
        input_shape = ctx.make_node('Shape', [node.input[0]])
        input_shape_size = ctx.make_node('Shape', [input_shape.output[0]])
        matrice_shape = ctx.make_node('Slice',
                                      [input_shape.output[0], const_negative_two.output[0], input_shape_size.output[0]])
        matrice_shape_float = ctx.make_node('Cast', [matrice_shape.output[0]], attr={'to': TensorProto.FLOAT})
        matrice_shape_float_x = ctx.make_node('Slice', [matrice_shape_float.output[0], const_zero.output[0],
                                                        const_one.output[0]])
        matrice_shape_float_y = ctx.make_node('Slice',
                                              [matrice_shape_float.output[0], const_one.output[0], const_two.output[0]])
        min_matrice_dim_float = ctx.make_node('Min', [matrice_shape_float_x.output[0], matrice_shape_float_y.output[0]])
        min_matrice_dim = ctx.make_node('Cast', [min_matrice_dim_float.output[0]], attr={'to': TensorProto.INT64})
        double_matrice_dim = ctx.make_node('Concat', [min_matrice_dim.output[0], min_matrice_dim.output[0]],
                                           attr={'axis': -1})
        sliced_input = ctx.make_node('Slice', [node.input[0], const_zero_zero.output[0], double_matrice_dim.output[0],
                                               const_negative_two_one.output[0]])
        sliced_input_shape = ctx.make_node('Shape', [sliced_input.output[0]])
        sliced_input_shape_half = ctx.make_node('Slice', [sliced_input_shape.output[0], const_zero.output[0],
                                                          const_negative_one.output[0]])
        sliced_input_shape_new = ctx.make_node('Concat', [sliced_input_shape_half.output[0], const_one.output[0]],
                                               attr={'axis': -1})
        matrice_range = ctx.make_node('Range', [const_zero.output[0], min_matrice_dim.output[0], const_one.output[0]])
        unsqueezed_matrice_range = ctx.make_node('Unsqueeze', [matrice_range.output[0]], attr={"axes": [-1]})
        expanded_range = ctx.make_node('Expand', [unsqueezed_matrice_range.output[0], sliced_input_shape_new.output[0]])
        gathered_result = ctx.make_node('GatherElements', [sliced_input.output[0], expanded_range.output[0]],
                                        attr={'axis': -1})
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        squeezed_result = ctx.make_node('Squeeze', [gathered_result.output[0]], attr={"axes": [-1]},
                                        name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)
