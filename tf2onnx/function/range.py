# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - range op conversion
"""
import numpy as np
from onnx.onnx_pb import TensorProto
from tf2onnx import utils

# pylint: disable=unused-argument,missing-docstring


def make_range_const(ctx, start, limit, delta, output, scope_name, dtype):
    """make Range subgraph if all inputs are const."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(scope_name)
    start = ctx.get_node_by_output(start).get_tensor_value(as_list=False)
    limit = ctx.get_node_by_output(limit).get_tensor_value(as_list=False)
    delta = ctx.get_node_by_output(delta).get_tensor_value(as_list=False)
    val = np.arange(start, limit, delta, dtype=start.dtype)
    const_range = ctx.make_const(base_name, val)
    return [ctx.make_node("Identity", [const_range.output[0]], dtypes=[dtype], outputs=[output]),
            const_range]


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
    nodes.append(ctx.make_const(cond_name, np.ones((), dtype=bool)))

    # body
    g = ctx.create_new_graph_with_same_config()
    body_nodes = [g.make_node("Identity", ["cond"], outputs=["cond_out"]),
                  g.make_node("Add", ["prev", delta], outputs=["current"], name=utils.make_name("add")),
                  g.make_node("Identity", ["prev"], outputs=["range"])]
    g.set_nodes(body_nodes)
    g.add_graph_input("i", TensorProto.INT64, [])
    g.add_graph_input("cond", TensorProto.BOOL, [])
    g.add_graph_input("prev", dtype, [])

    g.add_graph_output("cond_out", TensorProto.BOOL, [])
    g.add_graph_output("current", dtype, [])
    g.add_graph_output("range", dtype, [])

    # loop
    loop_inputs = [trip_count_node.output[0], cond_name, start]
    loop_node = ctx.make_node("Loop", loop_inputs, output_count=2, op_name_scope=base_name, name="loop")
    loop_node.set_body_graph_as_attr("body", g)
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
    utils.make_sure(dtype is not None, "Tidx of %s is None", node.name)
    return make_range(ctx, node.input[0], node.input[1], node.input[2],
                      node.output[0], name, dtype)
