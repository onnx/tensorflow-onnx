# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - select op conversion
"""
import numpy as np
from onnx import helper, onnx_pb
from onnx.onnx_pb import TensorProto
from tf2onnx import utils
from tf2onnx.graph import Node
from tf2onnx.utils import port_name, make_sure


# pylint: disable=useless-return,broad-except,logging-not-lazy,unused-argument,missing-docstring


def select_op8(ctx, node, name, args):
    # T output = Select(bool condition, T x, T y)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    utils.make_sure(len(node.input) > 1, "Select with only condition is not supported.")

    nodes = []
    true_data_type = ctx.get_dtype(node.input[1])
    false_data_type = ctx.get_dtype(node.input[2])
    true_data_shape = ctx.get_shape(node.input[1])
    false_data_shape = ctx.get_shape(node.input[2])
    make_sure(true_data_type == false_data_type, "select true val and false val have different data types.")
    make_sure(np.array_equal(true_data_shape, false_data_shape),
              "select true val and false val have different output shapes.")

    condition_shape = ctx.get_shape(node.input[0])
    utils.make_sure(condition_shape is not None, "condition shape is None")
    rank = len(condition_shape)

    utils.make_sure(rank >= 0, "rank should be >= 0")
    val_output_id = None
    if rank > 0:
        # create nodes getting shape of condition
        shape_node_output_shape = [rank]
        shape_node = ctx.make_node("Shape", [node.input[0]], op_name_scope=node.name,
                                   shapes=[shape_node_output_shape], dtypes=[onnx_pb.TensorProto.INT64])
        nodes.append(shape_node)

        # todo(pengwa), move those leveraging rewrite_incomplete_type_support_onnxruntime after shape inferencing
        # bug is fixed.
        # workaround: onnxruntime does not support Split-2, add cases before and after.
        target_dtype = onnx_pb.TensorProto.FLOAT
        shape_f_node = ctx.make_node("Cast", [shape_node.output[0]], attr={"to": target_dtype},
                                     shapes=[shape_node_output_shape], dtypes=[target_dtype],
                                     op_name_scope=node.name)
        nodes.append(shape_f_node)

        split_attr = [1 for i in range(rank)]
        output_shapes = [[1] for i in range(rank)]
        output_dtypes = [target_dtype for i in range(rank)]
        split_node = ctx.make_node("Split", [shape_f_node.output[0]], output_count=rank,
                                   attr={"split": split_attr}, shapes=output_shapes,
                                   dtypes=output_dtypes, op_name_scope=node.name)
        nodes.append(split_node)

        trip_cnts = []
        for i in range(rank):
            output_id = split_node.output[i]
            output_shape = ctx.get_shape(output_id)
            target_dtype = onnx_pb.TensorProto.INT64
            shape_i_node = ctx.make_node("Cast", [output_id], attr={"to": target_dtype},
                                         shapes=[output_shape], dtypes=[target_dtype],
                                         op_name_scope=node.name)
            trip_cnts.append(shape_i_node.output[0])
            nodes.append(shape_i_node)
        # workaround ends

        onnx_nodes = create_loop_op(node.input, true_data_type, true_data_shape, trip_cnts, rank)
        new_nodes = [Node(n, ctx, skip_conversion=True) for n in onnx_nodes]
        nodes.extend(new_nodes)
        loop_node = new_nodes[-1]
        val_output_id = loop_node.output[1]
    elif rank == 0:
        if_onnx_node, val_output_id = create_if_op(node.input, true_data_type, true_data_shape)
        if_node = Node(if_onnx_node, ctx, skip_conversion=True)
        nodes.append(if_node)

    ctx.copy_shape(node.output[0], val_output_id)
    ctx.set_dtype(node.output[0], true_data_type)

    output_node = ctx.make_node("Identity", [val_output_id], name=node.name,
                                shapes=[ctx.get_shape(val_output_id)], dtypes=[true_data_type])
    nodes.append(output_node)

    return nodes


# gather_input_ids is 1-D tensor, containing 3 elements:
# 0: condition data to gather on
# 1: true result to gather on
# 2: false result to father on
def create_loop_op(gather_input_ids, output_type, output_shape, trip_count_input_ids, rank):
    nodes = []

    cond_var_name = utils.make_name("condition")
    true = helper.make_tensor(cond_var_name, TensorProto.BOOL, (), [True])
    init_cond = helper.make_node("Constant", [], [cond_var_name], value=true, name=cond_var_name)
    nodes.append(init_cond)

    # Loop requires at least a variable, add a useless fake variable.
    fake_val_name = utils.make_name("fake_var")
    fake_var_init_val = helper.make_tensor(fake_val_name, TensorProto.FLOAT, (), [0.0])
    fake_var_init_node = helper.make_node("Constant", [], [fake_val_name],
                                          value=fake_var_init_val, name=fake_val_name)
    nodes.append(fake_var_init_node)

    if rank < 1:
        raise ValueError("rank is < 1")
    trip_count_input_id = trip_count_input_ids[-1 * rank]

    op_name = utils.make_name("loop")
    fake_var_output_id = port_name(op_name)
    loop_inputs = [trip_count_input_id,  # trip count
                   cond_var_name,  # termination condition
                   fake_val_name  # initial value of loop-carried dependencies
                  ]
    loop_scan_output_id = port_name(op_name, 1)

    loop_body = create_loop_body_graph(gather_input_ids, output_type, output_shape, trip_count_input_ids, rank, op_name)
    loop_node = helper.make_node("Loop", loop_inputs, [fake_var_output_id, loop_scan_output_id],
                                 name=op_name, body=loop_body)
    nodes.append(loop_node)
    return nodes


def get_inputs_for_current_iteration(input_id, iter_index):
    nodes = []
    op_name = utils.make_name("Gather")
    cond_gather_out_name = port_name(op_name)
    cond_gather_node = helper.make_node("Gather", [input_id, iter_index], [cond_gather_out_name],
                                        name=op_name)
    nodes.append(cond_gather_node)

    op_name = utils.make_name("Squeeze")
    cur_cond_val_out_name = port_name(op_name)
    cur_cond_val_scalar_node = helper.make_node("Squeeze", [cond_gather_out_name], [cur_cond_val_out_name],
                                                name=op_name, axes=[0])
    nodes.append(cur_cond_val_scalar_node)

    return nodes, cur_cond_val_out_name


def create_loop_body_graph(gather_input_ids, output_data_type, output_shape, trip_count_input_ids, rank, loop_name):
    nodes = []
    iter_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    fake_var_name = utils.make_name("fake_var")
    graph_inputs = [helper.make_tensor_value_info(iter_name, TensorProto.INT64, (1,)),  # iteration_num
                    helper.make_tensor_value_info(cond_name, TensorProto.BOOL, ()),  # condition
                    helper.make_tensor_value_info(fake_var_name, TensorProto.FLOAT, ())  # loop-carried dependency
                   ]

    # get the i'th value of condition
    cond_input_id = gather_input_ids[0]
    new_nodes, cond_input_id_for_current_iter = get_inputs_for_current_iteration(cond_input_id, iter_name)
    nodes.extend(new_nodes)

    # get the i'th value of true values
    true_input_id = gather_input_ids[1]
    new_nodes, true_input_id_for_current_iter = get_inputs_for_current_iteration(true_input_id, iter_name)
    nodes.extend(new_nodes)


    # get the i'th value of false values
    false_input_id = gather_input_ids[2]
    new_nodes, false_input_id_for_current_iter = get_inputs_for_current_iteration(false_input_id, iter_name)
    nodes.extend(new_nodes)

    input_ids_for_current_iter = [cond_input_id_for_current_iter, true_input_id_for_current_iter,
                                  false_input_id_for_current_iter]
    output_id = None
    rank = rank - 1
    if rank >= 1:
        nodes_1 = create_loop_op(input_ids_for_current_iter, output_data_type, output_shape[1:],
                                 trip_count_input_ids, rank)
        loop_1 = nodes_1[-1]
        output_id = loop_1.output[1]
        nodes.extend(nodes_1)
    elif rank == 0:
        if_node, if_node_output_id = create_if_op(input_ids_for_current_iter, output_data_type, output_shape[1:])
        output_id = if_node_output_id
        nodes.append(if_node)

    output_identity_name = utils.make_name("loop_output")
    loop_output_id = utils.port_name(output_identity_name)
    loop_output_node = helper.make_node(
        'Identity',
        [output_id],
        [loop_output_id],
        name=output_identity_name
    )
    nodes.append(loop_output_node)

    cond_identity_name = utils.make_name("cond_output")
    cond_output_id = utils.port_name(cond_identity_name)
    identity_node = helper.make_node(
        'Identity',
        [cond_name],
        [cond_output_id],
        name=cond_identity_name
    )
    nodes.append(identity_node)

    fake_var_identity_name = utils.make_name("fake_var_output")
    fake_var_output_id = utils.port_name(fake_var_identity_name)
    identity_node = helper.make_node(
        'Identity',
        [fake_var_name],
        [fake_var_output_id],
        name=fake_var_identity_name
    )
    nodes.append(identity_node)

    graph_outputs = [helper.make_tensor_value_info(cond_output_id, TensorProto.BOOL, ()),
                     helper.make_tensor_value_info(fake_var_output_id, TensorProto.FLOAT, ()),
                     helper.make_tensor_value_info(loop_output_id, output_data_type, output_shape[1:])]

    body_graph = helper.make_graph(nodes, utils.make_name(loop_name + "-body-graph"), graph_inputs,
                                   graph_outputs)
    return body_graph


def create_if_op(input_ids, output_data_type, output_shape):
    op_name = utils.make_name("If")
    true_graph = create_body_graph_for_if_branch(output_data_type, output_shape, input_ids[1], op_name)
    false_graph = create_body_graph_for_if_branch(output_data_type, output_shape, input_ids[2], op_name)
    out_name = port_name(op_name)

    # output a scalar
    if_node = helper.make_node("If", [input_ids[0]], [out_name], name=op_name, then_branch=true_graph,
                               else_branch=false_graph)
    return if_node, out_name


def create_body_graph_for_if_branch(data_type, output_shape, chosen_cur_cond_val_out_name, op_name):
    nodes = []

    name = utils.make_name("Identity")
    identity_node = helper.make_node(
        'Identity',
        [chosen_cur_cond_val_out_name],
        ['y'],
        name=name
    )
    nodes.append(identity_node)

    # create one output
    y = helper.make_tensor_value_info('y', data_type, output_shape)

    graph_def = helper.make_graph(
        nodes,
        utils.make_name(op_name +'-body-graph'),
        [],
        [y],
    )
    return graph_def
