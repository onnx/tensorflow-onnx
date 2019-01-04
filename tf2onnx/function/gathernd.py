# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - gathernd op conversion
"""
import numpy as np
from onnx import helper
from onnx.onnx_pb import TensorProto
from tf2onnx import utils
from tf2onnx.utils import make_onnx_inputs_outputs

# pylint: disable=unused-argument,missing-docstring

INT64_MAX = np.iinfo(np.int64).max

def _make_gathernd_inner_loop(ctx, params, index, dtype):
    """create the inner loop for GatherNd."""
    # gather_cur = params
    # for (int i=0; i<size(index); i++)
    #   gather_res = gather(gather_cur, index[i])
    scope_name = utils.make_name("gathernd_inner_loop")
    nodes = []
    trip_node = ctx.make_node("Size", [index.output[0]])
    nodes.append(trip_node.op)
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=np.bool))
    trip_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    cond_out_name = utils.make_name("cond_out")
    cur_name = utils.make_name("gather_cur")
    result_name = utils.make_name("res")
    body_inputs = [make_onnx_inputs_outputs(trip_name, TensorProto.INT64, []),
                   make_onnx_inputs_outputs(cond_name, TensorProto.BOOL, []),
                   make_onnx_inputs_outputs(cur_name, dtype, [])]
    body_outputs = [make_onnx_inputs_outputs(cond_out_name, TensorProto.BOOL, [],),
                    make_onnx_inputs_outputs(result_name, dtype, [])]
    body_nodes = []
    index_i = ctx.make_node("Gather", [index.output[0], trip_name], attr={"axis": 0})
    gather = ctx.make_node("Gather", [cur_name, index_i.output[0]], attr={"axis": 0})
    squeeze = ctx.make_node("Squeeze", [gather.output[0]], attr={"axes": [0]}, outputs=[result_name])
    body_nodes.extend([index_i.op, gather.op, squeeze.op,
                       utils.make_onnx_identity(cond_name, cond_out_name)])
    body_graph = helper.make_graph(body_nodes, utils.make_name("gathernd_inner_body"), body_inputs, body_outputs)
    inner_loop = ctx.make_node("Loop", [trip_node.output[0],
                                        cond_const.output[0],
                                        params],
                               op_name_scope=scope_name,
                               attr={"body": body_graph})
    nodes.append(inner_loop.op)
    return nodes, inner_loop


def make_gathernd(ctx, params, indices, output, scope_name, t_params):
    """make GatherNd op."""
    # Tparams output = GatherNd(Tparams params, Tidx indices)
    scope_name = utils.make_name(scope_name)
    nodes = []
    # reshape indices into [sum(indices[:-1]), indices[-1]]
    indices_shape = ctx.make_node("Shape", [indices], dtypes=[TensorProto.INT64])
    outter_shape = ctx.make_node("Slice",
                                 [indices_shape.output[0]],
                                 attr={"axes": [0], "ends": [-1], "starts": [0]},
                                 dtypes=[TensorProto.INT64])
    inner_shape = ctx.make_node("Slice",
                                [indices_shape.output[0]],
                                attr={"axes": [0], "ends": [INT64_MAX], "starts": [-1]},
                                dtypes=[TensorProto.INT64])
    outter_shape_sum = ctx.make_node("ReduceSum",
                                     [outter_shape.output[0]],
                                     attr={"axes": [0], "keepdims": 1},
                                     dtypes=[TensorProto.INT64])
    flatten_shape = ctx.make_node("Concat",
                                  [outter_shape_sum.output[0], inner_shape.output[0]],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    flatten_indices = ctx.make_node("Reshape", [indices, flatten_shape.output[0]])
    nodes.extend([indices_shape, outter_shape, inner_shape, outter_shape_sum, flatten_shape, flatten_indices])
    # outter loop for each index
    # for (int i=0; i<outter_shape_sum; i++) inner_loop(params, flatten_indices[i])
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=np.bool))
    dummy_const = ctx.make_const(utils.make_name("dummy"), np.ones((), dtype=np.int64))
    trip_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    cond_out_name = utils.make_name("cond_out")
    dummy_name = utils.make_name("dummy")
    dummy_out_name = utils.make_name("dummy_out")
    result_name = utils.make_name("res")
    body_inputs = [make_onnx_inputs_outputs(trip_name, TensorProto.INT64, []),
                   make_onnx_inputs_outputs(cond_name, TensorProto.BOOL, []),
                   make_onnx_inputs_outputs(dummy_name, t_params, [])]
    body_outputs = [make_onnx_inputs_outputs(cond_out_name, TensorProto.BOOL, [],),
                    make_onnx_inputs_outputs(dummy_out_name, t_params, []),
                    make_onnx_inputs_outputs(result_name, t_params, [])]
    body_nodes = []
    index = ctx.make_node("Gather", [flatten_indices.output[0], trip_name], attr={"axis": 0})
    index_squeeze = ctx.make_node("Squeeze", [index.output[0]], attr={"axes": [0]})
    # inner loop to gather result
    inner_loop_nodes, inner_loop = _make_gathernd_inner_loop(ctx,
                                                             params,
                                                             index_squeeze,
                                                             t_params)
    body_nodes.extend([index.op, index_squeeze.op] + inner_loop_nodes +
                      [utils.make_onnx_identity(cond_name, cond_out_name),
                       utils.make_onnx_identity(dummy_name, dummy_out_name),
                       utils.make_onnx_identity(inner_loop.output[0], result_name)])
    body_graph = helper.make_graph(body_nodes, utils.make_name("gathernd_body"), body_inputs, body_outputs)
    gathernd_loop = ctx.make_node("Loop",
                                  [outter_shape_sum.output[0], cond_const.output[0], params],
                                  output_count=2,
                                  op_name_scope=scope_name,
                                  attr={"body": body_graph})
    nodes.append(gathernd_loop)
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
    output_shape_ = ctx.make_node("Concat",
                                  [outter_shape.output[0], output_inner_shape.output[0]],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    output_shape = ctx.make_node("Slice",
                                 [output_shape_.output[0]],
                                 attr={"axes": [0], "ends": [-1], "starts": [0]},
                                 dtypes=[TensorProto.INT64])
    output_reshape = ctx.make_node("Reshape",
                                   [gathernd_loop.output[1], output_shape.output[0]],
                                   outputs=[output])
    nodes.extend([inner_loop_shape,
                  inner_loop_shape_,
                  output_inner_shape,
                  output_shape_,
                  output_shape,
                  output_reshape])
    return nodes


def gathernd_op(ctx, node, name, args):
    """GatherNd op."""
    # Tparams output = GatherNd(Tparams params, Tidx indices)
    params = node.input[0]
    indices = node.input[1]
    output = node.output[0]
    # same as the attr Tparams
    t_params = ctx.get_dtype(params)
    utils.make_sure(t_params, "Dtype of {} is None".format(indices))
    return make_gathernd(ctx, params, indices, output, name, t_params)
