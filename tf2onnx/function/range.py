# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - range op conversion
"""
import numpy as np
from onnx import helper, onnx_pb
from onnx.onnx_pb import TensorProto
from tf2onnx import utils

# pylint: disable=useless-return,broad-except,logging-not-lazy,unused-argument,missing-docstring
def make_range_const(ctx,
                     start_node,
                     limit_node,
                     delta_node,
                     output,
                     name,
                     dtype):
    """make Range subgraph if all inputs are const."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(name)
    start = start_node.get_tensor()
    limit = limit_node.get_tensor()
    delta = delta_node.get_tensor()
    val = np.arange(start, limit, delta, dtype=start.dtype)
    const_range = ctx.make_const(base_name, val)
    return ctx.make_node("Identity", [const_range.output[0]], dtypes=[dtype], outputs=[output])

def make_range_subgraph(ctx, start, limit, delta, output, name, dtype):
    """make Range subgraph."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(name)

    nodes = []

    # trip_count
    diff_node = ctx.make_node("Sub",
                              [limit, start],
                              op_name_scope=base_name,
                              name=utils.make_name("diff"))
    diff_output = diff_node.output[0]
    nodes.append(diff_node)

    delta_cast = delta
    if dtype in [onnx_pb.TensorProto.INT32, onnx_pb.TensorProto.INT64]:
        cast_node = ctx.make_node("Cast", [diff_output], op_name_scope=base_name,
                                  name="cast_diff", attr={"to": onnx_pb.TensorProto.FLOAT})
        nodes.append(cast_node)
        diff_output = cast_node.output[0]

        cast_node = ctx.make_node("Cast", [delta], op_name_scope=base_name, name="cast_delta",
                                  attr={"to": onnx_pb.TensorProto.FLOAT})
        nodes.append(cast_node)
        delta_cast = cast_node.output[0]

    div_node = ctx.make_node("Div", [diff_output, delta_cast], op_name_scope=base_name, name="div")
    nodes.append(div_node)

    ceil_node = ctx.make_node("Ceil", [div_node.output[0]], op_name_scope=base_name, name="ceil")
    nodes.append(ceil_node)

    trip_count_node = ctx.make_node("Cast", [ceil_node.output[0]], op_name_scope=base_name, name="trip_cnt",
                                    attr={"to": onnx_pb.TensorProto.INT64})
    nodes.append(trip_count_node)

    # cond
    # Use initializer here since Constant OP before opset 9 does not support bool type
    cond_name = "{}_cond".format(base_name)
    ctx.make_const(cond_name, np.ones((), dtype=bool))

    # body
    body_inputs = [utils.make_onnx_inputs_outputs("i", onnx_pb.TensorProto.INT64, []),
                   utils.make_onnx_inputs_outputs("cond", onnx_pb.TensorProto.BOOL, []),
                   utils.make_onnx_inputs_outputs("prev", dtype, [])]
    body_outputs = [utils.make_onnx_inputs_outputs("cond_out", onnx_pb.TensorProto.BOOL, []),
                    utils.make_onnx_inputs_outputs("current", dtype, []),
                    utils.make_onnx_inputs_outputs("range", dtype, [])]
    body_nodes = []
    body_nodes.append(utils.make_onnx_identity("cond", "cond_out"))
    body_nodes.append(helper.make_node("Add", ["prev", delta], ["current"], name=utils.make_name("add")))
    body_nodes.append(utils.make_onnx_identity("prev", "range"))
    body_graph = helper.make_graph(body_nodes, utils.make_name("{}_body".format(base_name)), body_inputs, body_outputs)

    # loop
    loop_inputs = [trip_count_node.output[0], cond_name, start]
    loop_node = ctx.make_node("Loop", loop_inputs, output_count=2, op_name_scope=base_name, name="loop",
                              attr={"body": body_graph})
    nodes.append(loop_node)

    identity_node = ctx.make_node("Identity", [loop_node.output[1]], name=base_name, dtypes=[dtype], outputs=[output])
    nodes.append(identity_node)

    return nodes

def range_op7(ctx, node, name, args):
    """Range."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    start_node = node.inputs[0]
    limit_node = node.inputs[1]
    delta_node = node.inputs[2]

    output_name = node.output[0]
    dtype = node.get_attr_int("Tidx")
    if start_node.is_const() and limit_node.is_const() and delta_node.is_const():
        return make_range_const(ctx,
                                start_node,
                                limit_node,
                                delta_node,
                                output_name,
                                name,
                                dtype)
    return make_range_subgraph(ctx,
                               start_node.output[0],
                               limit_node.output[0],
                               delta_node.output[0],
                               output_name,
                               name,
                               dtype)
