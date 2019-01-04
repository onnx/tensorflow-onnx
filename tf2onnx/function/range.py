# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - range op conversion
"""
import numpy as np
from onnx import helper
from onnx.onnx_pb import TensorProto
from tf2onnx import utils

# pylint: disable=unused-argument,missing-docstring
def make_range_const(ctx, start, limit, delta, output, scope_name, dtype):
    """make Range subgraph if all inputs are const."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(scope_name)
    start = ctx.get_node_by_output(start).get_tensor()
    limit = ctx.get_node_by_output(limit).get_tensor()
    delta = ctx.get_node_by_output(delta).get_tensor()
    val = np.arange(start, limit, delta, dtype=start.dtype)
    const_range = ctx.make_const(base_name, val)
    return ctx.make_node("Identity", [const_range.output[0]], dtypes=[dtype], outputs=[output])


def make_range_non_const(ctx, start, limit, delta, output, scope_name, dtype):
    """make Range subgraph."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(scope_name)

    nodes = []

    # trip_count
    diff_node = ctx.make_node("Sub",
                              [limit, start],
                              op_name_scope=base_name,
                              name=utils.make_name("diff"))
    diff_output = diff_node.output[0]
    nodes.append(diff_node)

    delta_cast = delta
    if dtype in [TensorProto.INT32, TensorProto.INT64]:
        cast_node = ctx.make_node("Cast", [diff_output], op_name_scope=base_name,
                                  name="cast_diff", attr={"to": TensorProto.FLOAT})
        nodes.append(cast_node)
        diff_output = cast_node.output[0]

        cast_node = ctx.make_node("Cast", [delta], op_name_scope=base_name, name="cast_delta",
                                  attr={"to": TensorProto.FLOAT})
        nodes.append(cast_node)
        delta_cast = cast_node.output[0]

    div_node = ctx.make_node("Div", [diff_output, delta_cast], op_name_scope=base_name, name="div")
    nodes.append(div_node)

    ceil_node = ctx.make_node("Ceil", [div_node.output[0]], op_name_scope=base_name, name="ceil")
    nodes.append(ceil_node)

    trip_count_node = ctx.make_node("Cast", [ceil_node.output[0]], op_name_scope=base_name, name="trip_cnt",
                                    attr={"to": TensorProto.INT64})
    nodes.append(trip_count_node)

    # cond
    # Use initializer here since Constant OP before opset 9 does not support bool type
    cond_name = "{}_cond".format(base_name)
    ctx.make_const(cond_name, np.ones((), dtype=bool))

    # body
    body_inputs = [utils.make_onnx_inputs_outputs("i", TensorProto.INT64, []),
                   utils.make_onnx_inputs_outputs("cond", TensorProto.BOOL, []),
                   utils.make_onnx_inputs_outputs("prev", dtype, [])]
    body_outputs = [utils.make_onnx_inputs_outputs("cond_out", TensorProto.BOOL, []),
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


def make_range(ctx, start, limit, delta, output, scope_name, dtype):
    if all(ctx.get_node_by_output(n).is_const() for n in [start, limit, delta]):
        return make_range_const(ctx, start, limit, delta, output, scope_name, dtype)
    return make_range_non_const(ctx, start, limit, delta, output, scope_name, dtype)


def range_op7(ctx, node, name, args):
    """Range."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    dtype = node.get_attr_int("Tidx")
    utils.make_sure(dtype, "Tidx of {} is None".format(node.name))
    return make_range(ctx, node.input[0], node.input[1], node.input[2],
                      node.output[0], name, dtype)
