# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - select op conversion
"""
from onnx import helper
from onnx.onnx_pb import TensorProto
from tf2onnx import utils
from tf2onnx.graph import Node
from tf2onnx.utils import port_name

# pylint: disable=useless-return,broad-except,logging-not-lazy,unused-argument,missing-docstring

def select_op(ctx, node, name, args):
    # T output = Select(bool condition, T x, T y)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    if len(node.input) == 1:
        raise ValueError("Select with only condition is not supported.")
    nodes = []
    data_type = ctx.get_dtype(node.input[1])

    op_name = utils.make_name("Size")
    out_name = port_name(op_name)
    batch_size_node = Node(helper.make_node("Size", [node.input[0]], [out_name], name=op_name), ctx)
    nodes.append(batch_size_node)

    nodes_to_append = create_loop_op(ctx, node, batch_size_node.output[0], data_type)
    nodes.extend(nodes_to_append)

    loop_scan_output_id = nodes[-1].output[1]
    ctx.copy_shape(node.output[0], loop_scan_output_id)
    ctx.set_dtype(node.output[0], data_type)
    ctx.replace_all_inputs(ctx.get_nodes(), node.output[0], loop_scan_output_id)
    return nodes


def create_loop_op(ctx, node, batch_val_input_id, data_type):
    nodes = []

    cond_var_name = "condition"
    true = helper.make_tensor(cond_var_name, TensorProto.BOOL, (), [True])
    init_cond = Node(helper.make_node("Constant", [], [cond_var_name], value=true), ctx)
    nodes.append(init_cond)

    # Loop requires at least a variable, add a useless fake variable.
    fake_val_name = "fake_var"
    fake_var_init_val = helper.make_tensor(fake_val_name, TensorProto.FLOAT, (), [0.0])
    fake_var_init_node = Node(helper.make_node("Constant", [], ["fake_var"], value=fake_var_init_val), ctx)
    nodes.append(fake_var_init_node)

    op_name = utils.make_name("Loop")
    out_name = port_name(op_name)
    loop_inputs = [batch_val_input_id, # trip count
                   cond_var_name, # termination condition
                   fake_val_name # initial value of loop-carried dependencies
                  ]
    loop_scan_output_id = port_name(op_name, 1)
    loop_node = Node(helper.make_node("Loop", loop_inputs, [out_name, loop_scan_output_id], name=op_name), ctx)
    loop_body = create_loop_body_graph(ctx, node, node.input[0], data_type)
    loop_node.set_attr("body", loop_body)
    ctx.add_body_graph(out_name, loop_body)
    nodes.append(loop_node)
    return nodes


def create_loop_body_graph(ctx, node, select_condition_input_id, select_output_data_type):
    nodes = []
    graph_inputs = [helper.make_tensor_value_info("i", TensorProto.INT64, ()), # iteration_num
                    helper.make_tensor_value_info("cond", TensorProto.BOOL, ()), # condition
                    helper.make_tensor_value_info("fake_var", TensorProto.FLOAT, ()) # loop-carried dependency
                   ]

    # get the i'th value of "Select"'s condition
    op_name = utils.make_name("Gather")
    cond_gather_out_name = port_name(op_name)
    cond_gather_node = helper.make_node("Gather", [select_condition_input_id, "i"], [cond_gather_out_name],
                                        name=op_name)
    nodes.append(cond_gather_node)

    op_name = utils.make_name("Squeeze")
    cur_cond_val_out_name = port_name(op_name)
    cur_cond_val_scalar_node = helper.make_node("Squeeze", [cond_gather_out_name], [cur_cond_val_out_name],
                                                name=op_name, axes=[0])
    nodes.append(cur_cond_val_scalar_node)

    if_node, if_node_output_id = create_if_op(ctx, node, cur_cond_val_out_name)
    nodes.append(if_node)

    identity_node = helper.make_node(
        'Identity',
        [if_node_output_id],
        ['output'],
    )
    nodes.append(identity_node)

    identity_node = helper.make_node(
        'Identity',
        ["cond"],
        ['cond_output'],
    )
    nodes.append(identity_node)

    identity_node = helper.make_node(
        'Identity',
        ["fake_var"],
        ['fake_var_output'],
    )
    nodes.append(identity_node)

    graph_outputs = [helper.make_tensor_value_info("cond_output", TensorProto.BOOL, ()),
                     helper.make_tensor_value_info("fake_var_output", TensorProto.FLOAT, ()),
                     helper.make_tensor_value_info("output", select_output_data_type, ())]

    body_graph = helper.make_graph(nodes, "loop_body_graph", graph_inputs, graph_outputs)
    return body_graph


def create_if_op(ctx, node, cur_cond_val_out_name):
    true_graph = create_body_graph_for_if_branch(ctx, node.input[1])
    false_graph = create_body_graph_for_if_branch(ctx, node.input[2])

    op_name = utils.make_name("If")
    out_name = port_name(op_name)

    # output a scalar
    if_node = helper.make_node("If", [cur_cond_val_out_name], [out_name], name=op_name,
                               then_branch=true_graph, else_branch=false_graph)
    ctx.add_body_graph(out_name, true_graph)
    ctx.add_body_graph(out_name, false_graph)
    return if_node, out_name


def create_body_graph_for_if_branch(ctx, input_id):
    data_type = ctx.get_dtype(input_id)
    nodes = []

    op_name = utils.make_name("Gather")
    true_out_name = port_name(op_name)
    true_gather_node = helper.make_node("Gather", [input_id, "i"], [true_out_name], name=op_name)
    nodes.append(true_gather_node)

    op_name = utils.make_name("Squeeze")
    true_squeeze_out_name = port_name(op_name)
    cur_true_val_scalar_node = helper.make_node("Squeeze", [true_out_name], [true_squeeze_out_name],
                                                name=op_name, axes=[0])
    nodes.append(cur_true_val_scalar_node)

    identity_node = helper.make_node(
        'Identity',
        [true_squeeze_out_name],
        ['y'],
    )
    nodes.append(identity_node)

    # create one output
    y = helper.make_tensor_value_info('y', data_type, ())

    graph_def = helper.make_graph(
        nodes,
        'if-body-graph',
        [],
        [y],
    )
    return graph_def
