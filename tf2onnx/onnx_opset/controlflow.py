# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
controlflow
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging

import numpy as np

from onnx import onnx_pb
from onnx.onnx_pb import TensorProto
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.utils import make_sure
from tf2onnx.tf_loader import find_function


logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

def get_inputs_for_current_iteration(g, input_id, iter_index):
    cond_gather_node = g.make_node("Gather", [input_id, iter_index])
    cur_cond_val_scalar_node = g.make_node("Squeeze", [cond_gather_node.output[0]], attr={"axes": [0]})
    return cur_cond_val_scalar_node.output[0]


def create_loop_body_graph(parent_g, gather_input_ids, output_data_type, output_shape, trip_count_input_ids,
                           rank, loop_name):
    g = parent_g.create_new_graph_with_same_config()
    g.parent_graph = parent_g
    iter_name = utils.make_name("i")
    cond_name = utils.make_name("cond")
    fake_var_name = utils.make_name("fake_var")

    g.add_graph_input(iter_name, TensorProto.INT64, (1,))  # iteration_num
    g.add_graph_input(cond_name, TensorProto.BOOL, ())  # condition
    g.add_graph_input(fake_var_name, TensorProto.FLOAT, ())  # loop-carried dependency

    # get the i'th value of condition
    cond_input_id = gather_input_ids[0]
    cond_input_id_for_current_iter = get_inputs_for_current_iteration(g, cond_input_id, iter_name)

    # get the i'th value of true values
    true_input_id = gather_input_ids[1]
    true_input_id_for_current_iter = get_inputs_for_current_iteration(g, true_input_id, iter_name)

    # get the i'th value of false values
    false_input_id = gather_input_ids[2]
    false_input_id_for_current_iter = get_inputs_for_current_iteration(g, false_input_id, iter_name)

    input_ids_for_current_iter = [cond_input_id_for_current_iter, true_input_id_for_current_iter,
                                  false_input_id_for_current_iter]
    output_id = None
    rank -= 1
    if rank >= 1:
        loop_1 = create_loop_op(g, input_ids_for_current_iter, output_data_type, output_shape[1:],
                                trip_count_input_ids, rank)
        output_id = loop_1.output[1]
    elif rank == 0:
        _, if_node_output_id = create_if_op(g, input_ids_for_current_iter, output_data_type, output_shape[1:])
        output_id = if_node_output_id

    output_identity_name = utils.make_name("loop_output")
    loop_output_id = utils.port_name(output_identity_name)
    g.make_node(
        'Identity',
        [output_id],
        outputs=[loop_output_id],
        name=output_identity_name
    )

    cond_identity_name = utils.make_name("cond_output")
    cond_output_id = utils.port_name(cond_identity_name)
    g.make_node(
        'Identity',
        [cond_name],
        outputs=[cond_output_id],
        name=cond_identity_name
    )

    fake_var_identity_name = utils.make_name("fake_var_output")
    fake_var_output_id = utils.port_name(fake_var_identity_name)
    g.make_node(
        'Identity',
        [fake_var_name],
        outputs=[fake_var_output_id],
        name=fake_var_identity_name
    )

    g.add_graph_output(cond_output_id, TensorProto.BOOL, ())
    g.add_graph_output(fake_var_output_id, TensorProto.FLOAT, ())

    # use None for all dims, just keep original rank. Because it is observed, dims might be changed in loop.
    g.add_graph_output(loop_output_id, output_data_type, utils.create_vague_shape_like(output_shape[1:]))

    return g


def create_if_op(g, input_ids, output_data_type, output_shape):
    op_name = utils.make_name("If")
    true_graph = create_body_graph_for_if_branch(g, output_data_type, output_shape, input_ids[1], op_name)
    false_graph = create_body_graph_for_if_branch(g, output_data_type, output_shape, input_ids[2], op_name)
    out_name = utils.port_name(op_name)

    # output a scalar
    if_node = g.make_node("If", [input_ids[0]], outputs=[out_name], name=op_name, skip_conversion=True)
    if_node.set_body_graph_as_attr("then_branch", true_graph)
    if_node.set_body_graph_as_attr("else_branch", false_graph)
    return if_node, out_name


def create_body_graph_for_if_branch(parent_g, data_type, output_shape, chosen_cur_cond_val_out_name, op_name):
    g = parent_g.create_new_graph_with_same_config()
    g.parent_graph = parent_g
    name = utils.make_name("Identity")
    g.make_node(
        'Identity',
        inputs=[chosen_cur_cond_val_out_name],
        outputs=['y'],
        name=name
    )
    g.add_graph_output("y", data_type, utils.create_vague_shape_like(output_shape))
    return g


# gather_input_ids is 1-D tensor, containing 3 elements:
# 0: condition data to gather on
# 1: true result to gather on
# 2: false result to gather on
def create_loop_op(g, gather_input_ids, output_type, output_shape, trip_count_input_ids, rank):
    cond_var_name = utils.make_name("cond_var")
    g.make_const(cond_var_name, np.array(True, dtype=np.bool))

    # Loop requires at least a variable, add a useless fake variable.
    fake_val_name = utils.make_name("fake_var")
    g.make_const(fake_val_name, np.array(0.0, dtype=np.float32))

    if rank < 1:
        raise ValueError("rank is < 1")
    trip_count_input_id = trip_count_input_ids[-1 * rank]

    loop_inputs = [trip_count_input_id,  # trip count
                   cond_var_name,  # termination condition
                   fake_val_name  # initial value of loop-carried dependencies
                   ]
    # define an extra scan output
    loop_node = g.make_node("Loop", loop_inputs, output_count=2, op_name_scope="select_loop",
                            skip_conversion=False)
    loop_body = create_loop_body_graph(g, gather_input_ids, output_type, output_shape, trip_count_input_ids,
                                       rank, loop_node.name)
    loop_node.set_body_graph_as_attr("body", loop_body)
    return loop_node


def make_range_const(ctx, start, limit, delta, output, scope_name, shape, dtype):
    """make Range subgraph if all inputs are const."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(scope_name)
    start = ctx.get_node_by_output(start).get_tensor_value(as_list=False)
    limit = ctx.get_node_by_output(limit).get_tensor_value(as_list=False)
    delta = ctx.get_node_by_output(delta).get_tensor_value(as_list=False)
    val = np.arange(start, limit, delta, dtype=start.dtype)
    const_range = ctx.make_const(base_name, val)
    ctx.make_node("Identity", [const_range.output[0]], shapes=[shape], dtypes=[dtype], outputs=[output])


def make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype):
    """make Range subgraph."""
    # T range = Range(T start, T limit, T delta)
    # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
    base_name = utils.make_name(scope_name)

    # trip_count
    diff_node = ctx.make_node("Sub",
                              [limit, start],
                              op_name_scope=base_name,
                              name=utils.make_name("diff"))
    diff_output = diff_node.output[0]

    delta_cast = delta
    if dtype in [TensorProto.INT32, TensorProto.INT64]:
        cast_node = ctx.make_node("Cast", [diff_output], op_name_scope=base_name,
                                  name="cast_diff", attr={"to": TensorProto.FLOAT})
        diff_output = cast_node.output[0]

        cast_node = ctx.make_node("Cast", [delta], op_name_scope=base_name, name="cast_delta",
                                  attr={"to": TensorProto.FLOAT})
        delta_cast = cast_node.output[0]
    div_node = ctx.make_node("Div", [diff_output, delta_cast], op_name_scope=base_name, name="div")
    ceil_node = ctx.make_node("Ceil", [div_node.output[0]], op_name_scope=base_name, name="ceil")
    trip_count_node = ctx.make_node("Cast", [ceil_node.output[0]], op_name_scope=base_name, name="trip_cnt",
                                    attr={"to": TensorProto.INT64})

    # cond
    # Use initializer here since Constant OP before opset 9 does not support bool type
    cond_name = "{}_cond".format(base_name)
    ctx.make_const(cond_name, np.ones((), dtype=bool))

    # body
    g = ctx.create_new_graph_with_same_config()
    g.parent_graph = ctx
    g.add_graph_input("i", TensorProto.INT64, [])
    g.add_graph_input("cond", TensorProto.BOOL, [])
    g.add_graph_input("prev", dtype, [])

    g.make_node("Identity", ["cond"], outputs=["cond_out"])
    g.make_node("Add", ["prev", delta], outputs=["current"], name=utils.make_name("add"))
    g.make_node("Identity", ["prev"], outputs=["range"])

    g.add_graph_output("cond_out", TensorProto.BOOL, [])
    g.add_graph_output("current", dtype, [])
    g.add_graph_output("range", dtype, [])

    # loop
    loop_inputs = [trip_count_node.output[0], cond_name, start]
    loop_node = ctx.make_node("Loop", loop_inputs, output_count=2, op_name_scope=base_name, name="loop")
    loop_node.set_body_graph_as_attr("body", g)

    ctx.make_node("Identity", [loop_node.output[1]], name=base_name, shapes=[shape], dtypes=[dtype], outputs=[output])


def make_range(ctx, start, limit, delta, output, scope_name, shape, dtype):
    if all(ctx.get_node_by_output(n).is_const() for n in [start, limit, delta]) is True:
        make_range_const(ctx, start, limit, delta, output, scope_name, shape, dtype)
    else:
        make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype)


@tf_op(["Loop", "Scan"])
class PassThroughOp:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change needed
        # loop has 1 less mandatory input
        # if = only doc changes
        # scan has 1 less mandatory input and 4 extra attrs
        pass


@tf_op("Range")
class Range:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        """Range."""
        # T range = Range(T start, T limit, T delta)
        # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
        dtype = node.get_attr_int("Tidx")
        shape = node.output_shapes[0]
        utils.make_sure(dtype is not None, "Tidx of %s is None", node.name)
        ctx.remove_node(node.name)
        make_range(ctx, node.input[0], node.input[1], node.input[2],
                   node.output[0], node.name, shape, dtype)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # opset 11 implements Range op explicitly
        pass


@tf_op(["Select", "SelectV2"])
class Select:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T output = Select(bool condition, T x, T y)
        # Select_res = Add(Multiply(Cast(bool condition, float32), T x,),
        #                  Multiply(Cast(Not(bool condition), float32), T y)).
        utils.make_sure(len(node.input) > 1, "Select with only condition is not supported.")
        positive_cast = ctx.make_node("Cast", [node.input[0]], name=utils.make_name(node.name),
                                      attr={"to": TensorProto.FLOAT})
        negative = ctx.make_node("Not", [node.input[0]], name=utils.make_name(node.name))
        negative_cast = ctx.make_node("Cast", [negative.output[0]], name=utils.make_name(node.name),
                                      attr={"to": TensorProto.FLOAT})
        multiply_1 = ctx.make_node("Mul", [positive_cast.output[0], node.input[1]], name=utils.make_name(node.name))
        multiply_2 = ctx.make_node("Mul", [node.input[2], negative_cast.output[0]], name=utils.make_name(node.name))
        add_name = node.name
        add_out = node.output
        dtype = ctx.get_dtype(node.output[0])
        shape = ctx.get_shape(node.output[0])
        ctx.remove_node(node.name)
        ctx.make_node("Add", [multiply_1.output[0], multiply_2.output[0]], outputs=add_out, name=add_name,
                      dtypes=[dtype], shapes=[shape])

    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        # T output = Select(bool condition, T x, T y)
        # V v_final_and_scan_outputs = Loop(int64 M, B cond, V v_initial)
        utils.make_sure(len(node.input) > 1, "Select with only condition is not supported.")

        true_data_type = ctx.get_dtype(node.input[1])
        true_data_shape = ctx.get_shape(node.input[1])
        make_sure(true_data_type is not None, "select true data dtype cannot be None")
        make_sure(true_data_shape is not None, "select true data shape cannot be None")

        condition_shape = ctx.get_shape(node.input[0])
        utils.make_sure(condition_shape is not None, "Shape of {} is None".format(node.input[0]))
        rank = len(condition_shape)

        utils.make_sure(rank >= 0, "rank should be >= 0")
        val_output_id = None
        if rank > 0:
            # create nodes getting shape of condition
            shape_node_output_shape = [rank]
            shape_node = ctx.make_node("Shape", [node.input[0]], op_name_scope=node.name,
                                       shapes=[shape_node_output_shape], dtypes=[TensorProto.INT64])

            # todo(pengwa), move those leveraging rewrite_incomplete_type_support_onnxruntime after shape inferencing
            # bug is fixed.
            # workaround: onnxruntime does not support Split-2, add cases before and after.
            target_dtype = TensorProto.FLOAT
            shape_f_node = ctx.make_node("Cast", [shape_node.output[0]], attr={"to": target_dtype},
                                         shapes=[shape_node_output_shape], dtypes=[target_dtype],
                                         op_name_scope=node.name)

            split_attr = [1 for i in range(rank)]
            output_shapes = [[1] for i in range(rank)]
            output_dtypes = [target_dtype for i in range(rank)]
            split_node = ctx.make_node("Split", [shape_f_node.output[0]], output_count=rank,
                                       attr={"split": split_attr}, shapes=output_shapes,
                                       dtypes=output_dtypes, op_name_scope=node.name)

            trip_cnts = []
            for i in range(rank):
                output_id = split_node.output[i]
                output_shape = ctx.get_shape(output_id)
                target_dtype = TensorProto.INT64
                shape_i_node = ctx.make_node("Cast", [output_id], attr={"to": target_dtype},
                                             shapes=[output_shape], dtypes=[target_dtype],
                                             op_name_scope=node.name)
                trip_cnts.append(shape_i_node.output[0])
            # workaround ends

            loop_node = create_loop_op(ctx, node.input, true_data_type, true_data_shape, trip_cnts, rank)

            val_output_id = loop_node.output[1]
        elif rank == 0:
            _, val_output_id = create_if_op(ctx, node.input, true_data_type, true_data_shape)

        ctx.copy_shape(node.output[0], val_output_id)
        ctx.set_dtype(node.output[0], true_data_type)
        ctx.remove_node(node.name)
        ctx.make_node("Identity", [val_output_id], outputs=node.output,
                      shapes=[ctx.get_shape(val_output_id)], dtypes=[true_data_type])

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T output = Select(bool condition, T x, T y)
        # T1 output = Where(bool condition, T1 x, T1 y)
        # NOTE: condition can be 1-dimension in tensorflow, while in onnx,
        # it should be broadcastable with other two inputs
        node.type = "Where"
        cond_shape = ctx.get_shape(node.input[0])
        make_sure(cond_shape is not None, "shape of {} is None".format(node.input[0]))
        input_shape = ctx.get_shape(node.input[1])
        if input_shape is None:
            input_shape = ctx.get_shape(node.input[2])
        make_sure(input_shape is not None, "input shape of {} is None".format(node.name))
        input_rank = len(input_shape)
        # if cond shape is 1-dimensional while input has higher rank, need to be reshaped to broadcast
        if len(cond_shape) == 1 and input_rank > 1:
            broadcast_shape = [cond_shape[0]] + [1] * (input_rank - 1)
            shape_const = ctx.make_const(utils.make_name(node.name), np.array(broadcast_shape, dtype=np.int64))
            reshape = ctx.make_node("Reshape", [node.input[0], shape_const.output[0]])
            ctx.replace_input(node, node.input[0], reshape.output[0], 0)


@tf_op("Where")
class Where:
    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # T_y output = Where(T_x condition), return indices of elements whose value are True
        node.type = "NonZero"
        # in onnx, indices are returned in this way [[ind_a_0, ind_b_0, ...], [ind_a_1, ind_b_1,...]];
        # while in tf, the result will be [[ind_a_0, ind_a_1, ...], [ind_b_0, ind_b_1, ...], ...]
        # this is the reason a transpose node inserted here.
        transpose_node = ctx.insert_new_node_on_output("Transpose",
                                                       node.output[0], name=utils.make_name("where_op_added"))
        ctx.copy_shape(node.output[0], transpose_node.output[0])
        ctx.copy_dtype(node.output[0], transpose_node.output[0])


@tf_op(["StatelessIf"])
class StatelessIfOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """V2 control flow - If"""
        inputs = node.input[1:]

        output_shapes = node.output_shapes
        output_dtypes = node.output_dtypes
        ctx.remove_node(node.name)

        # replace the original node
        if_node = ctx.make_node("If", node.input[:1], name=node.name, output_count=len(output_shapes),
                                shapes=output_shapes, dtypes=output_dtypes, skip_conversion=True)

        for branch in ["then_branch", "else_branch"]:
            func_name = node.get_attr_str(branch)
            g = find_function(func_name)
            g.parent_graph = ctx
            wire_if_branch(ctx, g, inputs, output_shapes, output_dtypes, func_name, node.name)
            if_node.set_body_graph_as_attr(branch, g)


@tf_op(["If"])
class IfOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """V2 control flow - If"""
        inputs = node.input[1:]

        if node.type == "If" and len(inputs) == 0:
            # this comes from the re-writers
            return

        output_shapes = node.output_shapes
        output_dtypes = node.output_dtypes
        ctx.remove_node(node.name)

        # replace the original node
        if_node = ctx.make_node("If", node.input[:1], name=node.name, output_count=len(output_shapes),
                                shapes=output_shapes, dtypes=output_dtypes, skip_conversion=True)

        for branch in ["then_branch", "else_branch"]:
            func_name = node.get_attr_str(branch)
            g = find_function(func_name)
            g.parent_graph = ctx
            wire_if_branch(ctx, g, inputs, output_shapes, output_dtypes, func_name, node.name)
            if_node.set_body_graph_as_attr(branch, g)


@tf_op(["TensorListSetItem"])
class TensorListSetItem:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # handled in 'While'
        pass


@tf_op(["TensorListGetItem"])
class TensorListGetItem:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        ctx.ta_reads.append(node.input[0])
        node.type = "Gather"
        ctx.replace_inputs(node, [node.input[0], node.input[1]])
        ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[1], name=node.child_name(), axes=[0])
        ctx.insert_new_node_on_output("Squeeze", node.output[0], name=node.child_name(), axes=[0])


@tf_op(["TensorListLength"])
class TensorListLength:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass


@tf_op(["TensorListReserve", "TensorListResize"])
class TensorListReserve:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        pass


@tf_op(["TensorListFromTensor"])
class TensorListFromTensor:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        consumers = ctx.find_output_consumers(node.output[0])
        if any([c.is_while() for c in consumers]):
            node.type = "Identity"
            ctx.copy_dtype(node.input[0], node.output[0])
            ctx.copy_shape(node.input[0], node.output[0])


@tf_op(["TensorListStack"])
class TensorListStack:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        if node.inputs[0].is_while():
            ctx.remove_node(node.name)
            ctx.replace_all_inputs(ctx.get_nodes(), node.output[0], node.input[0])


@tf_op(["While", "StatelessWhile"])
class While:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # the tensorflow while input is:
        #   loop_counter, max_iterations, [loop_vars]
        # cond and body use the same inputs
        # outputs are identical to inputs
        tf_while_inputs = node.input

        # the onnx loop input is:
        #   max_iterations, cond, [loop_vars]
        # body uses the inputs:
        #   iteration, cond, [loop_vars]
        # the onnx loop output is:
        #   cond [v_final_and_scan_outputs]

        output_shapes = node.output_shapes
        output_dtypes = node.output_dtypes
        # node.output must be copied as some element
        # may be removed from output_names below
        output_names = node.output.copy()

        # Make maximum_iterations int64 and replace -1(tf) with maxsize(onnx). If the const node has no other consumers,
        # modify it in place. Otherwise, make a new const node and leave the original unchanged.
        maximum_iterations_name = node.input[1]
        maximum_iterations = node.inputs[1].get_tensor_value()
        if maximum_iterations == -1:
            maximum_iterations = np.iinfo(np.int64).max
        consumers = ctx.find_output_consumers(maximum_iterations_name)
        external_consumers = [c for c in consumers if c != node and c.type != 'TensorListReserve']
        if len(external_consumers) == 0:
            ctx.remove_node(node.inputs[1].name)
        else:
            maximum_iterations_name = utils.make_name(node.inputs[1].name)
        ctx.make_const(maximum_iterations_name, np.array(maximum_iterations, dtype=np.int64))
        ctx.replace_input(node, node.input[1], maximum_iterations_name, 1)

        cond_name = node.get_attr_str("cond")
        cond_graph = find_function(cond_name)
        cond_graph.parent_graph = ctx

        body_name = node.get_attr_str("body")
        body = find_function(body_name)
        body.parent_graph = ctx

        loop_vars = [] # passed into the loop
        body_input_to_state_var = {} # Map from body input name to state var name
        cond_input_to_state_var = {}
        to_remove = []
        input_idx_to_remove = []
        # remove TensorListReserve
        for idx, name in enumerate(tf_while_inputs):
            if idx == 1:
                # onnx does not know maximum_iterations in the body so move this to a state var
                body_input_to_state_var[body.func_inputs[idx]] = maximum_iterations_name
                cond_input_to_state_var[cond_graph.func_inputs[idx]] = maximum_iterations_name
                continue
            if idx < 2:
                # skip  [0,1] loop_counter, max_iterations
                continue
            n = node.inputs[idx]
            if n.type in ["TensorListReserve", "TensorListResize"]:
                # there is no equivalent step in onnx and we should remove it.
                to_remove.append((idx, n))
                continue

            # tensor arrays we read from can't be loop_vars and we fetch them from the outer context instead
            if body.func_inputs[idx] in body.ta_reads:
                body_input_to_state_var[body.func_inputs[idx]] = name
                cond_input_to_state_var[cond_graph.func_inputs[idx]] = name
                input_idx_to_remove.append(idx)
            else:
                loop_vars.append(name)

        # loop_vars that become state_vars need to be removed from output as well
        for idx in reversed(input_idx_to_remove):
            del output_shapes[idx]
            del output_dtypes[idx]
            del output_names[idx]
            del body.outputs[idx]

        removed_scan_outputs = {}
        # remove tensor array that are passed in to the loop
        for idx, n in reversed(to_remove):
            ctx.remove_node(n.name)
            # make the node output bad
            ctx.replace_all_inputs(ctx.get_nodes(), n.output[0], "@@ALLOC")
            del body.func_inputs[idx]
            del cond_graph.func_inputs[idx]
            del tf_while_inputs[idx]
            # save the index of the scan output
            removed_scan_outputs[body.outputs[idx]] = idx
            del body.outputs[idx]
            # FIXME: Output shapes may be in wrong order if there are multiple scan outputs
            output_shapes.append(output_shapes[idx])
            output_dtypes.append(output_dtypes[idx])
            output_names.append(output_names[idx])
            del output_shapes[idx]
            del output_dtypes[idx]
            del output_names[idx]

        utils.make_sure(len(removed_scan_outputs) <= 1, "converter only supports while loops with a single scan output")

        ctx.remove_node(node.name)

        # In onnx 'cond' is a variable, not a function. We need to inject the subgraph into the main graph
        # before the loop and into the body.
        cond_binding = parameter_binding(cond_graph, tf_while_inputs)
        cond_outputs = inline_subgraph(ctx, cond_graph, cond_name, cond_binding)
        # onnx Loop op outputs only loop_vars so we need shift output dtypes/shapes and consumers
        output_shapes = output_shapes[2:]
        output_dtypes = output_dtypes[2:]
        output_names = output_names[2:]

        loop_node = ctx.make_node("Loop", [maximum_iterations_name, cond_outputs[0]] + loop_vars,
                                  output_count=len(output_shapes), name=node.name + "_loop",
                                  shapes=output_shapes, dtypes=output_dtypes, skip_conversion=True)

        output_map = dict(zip(output_names, loop_node.output))

        # shift output consumers
        for k, v in output_map.items():
            ctx.replace_all_inputs(ctx.get_nodes(), k, v)

        wire_while_body(ctx, body, loop_node.inputs, body_input_to_state_var, cond_input_to_state_var, output_shapes,
                        output_dtypes, body_name, node.name, cond_graph, tf_while_inputs, removed_scan_outputs)

        # if there was a tensorflow variant type, bind in a real type here
        # FIXME: I don't think this is needed anymore
        for i, n in enumerate(body.inputs):
            if body.get_dtype(n.output[0]) == onnx_pb.TensorProto.UNDEFINED:
                body.set_dtype(n.output[0], ctx.get_dtype(loop_node.input[i]))
        loop_node.set_body_graph_as_attr("body", body)


def wire_while_body(parent_g, g, loop_node_inputs, body_input_to_state_var, cond_input_to_state_var, output_shapes,
                    output_dtypes, scope, parent, cond_graph, tf_while_inputs, removed_scan_outputs):
    """Wire subgraph graph into main."""
    remove_parents = []
    to_remove = []

    # tensorflow function inputs that are state_vars come from outer context and
    # we need to remove them from the inputs by making the placeholder an identity
    for n in g.inputs:
        if n.output[0] in body_input_to_state_var:
            n.type = "Identity"
            g.replace_inputs(n, [body_input_to_state_var[n.output[0]]])

    # onnx will pass in cond as argument
    cond_node = g.make_node("Placeholder", [], name=utils.make_name("cond"),
                            output_count=1, dtypes=[onnx_pb.TensorProto.BOOL], shapes=[[]])

    # in onnx the body inputs are: index, cond, [loop_vars]
    func_inputs = [i for i in g.func_inputs[2:] if i not in body_input_to_state_var]
    func_inputs = [g.func_inputs[0], cond_node.output[0]] + func_inputs
    g.set_dtype(func_inputs[0], onnx_pb.TensorProto.INT64)
    g.func_inputs = func_inputs
    # tell graph lib to keep inputs in order
    g._order_sensitive_inputs = \
        [g.get_node_by_output(name) for name in func_inputs]  # pylint: disable=protected-access

    for p, c in zip(loop_node_inputs, func_inputs):
        shape = p.output_shapes[0]
        g.set_shape(c, shape)

    for i, node in enumerate(g.inputs):
        if node.output[0] not in func_inputs:
            remove_parents.append(node.output[0])

    # this is a tensor array write - make it an identity
    scan_outputs = []
    for node in g.get_nodes():
        if node.type == "TensorListSetItem":
            remove_parents.append(node.input[0])
            node.type = "Identity"
            g.set_shape(node.output[0], g.get_shape(node.input[2]))
            g.set_dtype(node.output[0], g.get_dtype(node.input[2]))
            g.replace_inputs(node, [node.input[2]])
            scan_outputs.append(node.output[0])

    if len(scan_outputs) != len(removed_scan_outputs):
        raise ValueError("While loop couldn't find scan output index for nodes")

    for output in scan_outputs:
        last_output = output
        consumers = g.find_output_consumers(last_output)
        while consumers:
            node = consumers[0]
            if node.type != "Identity":
                raise ValueError("While loop couldn't find scan output index for node " + node.name)
            last_output = node.output[0]
            consumers = g.find_output_consumers(last_output)
        if last_output not in removed_scan_outputs:
            raise ValueError("While loop couldn't find scan output index for node " + node.name)
        # TODO: store index to ensure scan outputs are in correct order for multiple outputs
        # initial_output_index = removed_scan_outputs[last_output]

    # remove all nodes feeding to TensorListSetItem's reserved tensor
    while remove_parents:
        output_name = remove_parents[0]
        del remove_parents[0]
        node = g.get_node_by_output(output_name)
        if node:
            if output_name not in func_inputs:
                if node.input:
                    remove_parents.extend(node.input)
                g.remove_node(node.name)

    for node in to_remove:
        g.remove_node(node.name)

    cond_binding = parameter_binding(cond_graph, func_inputs[:1] + g.outputs[2:], cond_input_to_state_var)
    cond_outputs = inline_subgraph(g, cond_graph, "cond__", cond_binding)

    g.outputs = [cond_outputs[0]] + g.outputs[2:] + scan_outputs

    # FIXME: onnx does not have a variant type so we try to fish for the dtype in a prior TensorListSetItem.
    for o in g.outputs:
        if g.get_dtype(o) == onnx_pb.TensorProto.UNDEFINED:
            node = g.get_node_by_output(o)
            if node.type in ["Identity"]:
                g.set_dtype(o, node.inputs[0].output_dtypes[0])

    return g


def wire_if_branch(parent_g, g, inputs, output_shapes, output_dtypes, scope, parent):
    """Wire subgraph graph into main."""
    binding = parameter_binding(g, inputs)
    to_remove = []
    for node in g.inputs:
        parent_name = binding.get(node.output[0])
        if parent_name and parent_name != "@@ALLOC":
            g.replace_inputs(node, [parent_name])
            node.type = "Identity"
        else:
            to_remove.append(node)

    for node in to_remove:
        g.remove_node(node.name)

    prefix_graph(g, scope)

    for shape, dtype, output_name in zip(output_shapes, output_dtypes, g.outputs):
        g.set_shape(output_name, shape)
        g.set_dtype(output_name, dtype)

    return g


def inline_subgraph(parent, g, scope, binding):
    # make a copy since we don't want to change the origianl graph
    g = copy.deepcopy(g)
    to_remove = []
    for node in g.inputs:
        parent_name = binding.get(node.output[0])
        if parent_name and parent_name != "@@ALLOC":
            g.replace_inputs(node, [parent_name])
            node.type = "Identity"
        else:
            to_remove.append(node)
    for node in to_remove:
        g.remove_node(node.name)
    prefix_graph(g, scope)
    for n in g.get_nodes():
        dtypes = n.output_dtypes
        shapes = n.output_shapes
        n.graph = parent
        for name, shape, dtype in zip(n.output, shapes, dtypes):
            # FIXME: don't access this directly
            parent._output_shapes[name] = shape  # pylint: disable=protected-access
            parent._dtypes[name] = dtype  # pylint: disable=protected-access

    ops = parent.get_nodes() + g.get_nodes()
    parent.reset_nodes(ops)

    # copy output shape and dtype to parent graph
    for name in g.outputs:
        parent.set_dtype(name, g.get_dtype(name))
        parent.set_shape(name, g.get_shape(name))

    return  g.outputs


def parameter_binding(g, inputs, state_vars=None):
    binding = {}
    i = 0
    for k in g.func_inputs:
        if state_vars and k in state_vars:
            binding[k] = state_vars[k]
        else:
            binding[k] = inputs[i]
            i += 1
    return binding


def prefix_graph(g, scope):
    ops = g.get_nodes()[:]
    to_remove = []
    for node in ops:
        output_shapes = node.output_shapes
        output_dtypes = node.output_dtypes
        attr = node.attr
        if node.is_graph_input():
            continue
        new_node = g.make_node(node.type, node.input, name=node.name, output_count=len(node.output),
                               shapes=output_shapes, dtypes=output_dtypes, attr=attr,
                               op_name_scope=scope, skip_conversion=True)
        attr_graphs = node.get_body_graphs()
        if attr_graphs:
            for k, v in attr_graphs.items():
                new_node.set_body_graph_as_attr(k, v)
        for old_output, new_output in zip(node.output, new_node.output):
            for i, oname in enumerate(g.outputs):
                if old_output == oname:
                    g.outputs[i] = new_output
                    break
            g.replace_all_inputs(ops, old_output, new_output)
        to_remove.append(node)
    for node in to_remove:
        g.remove_node(node.name)


def dump_graph(g):
    print()
    print("--, graph=", g.graph_name)
    t = ["{} {}/{}".format(n.name, g.get_shape(n.output[0]), g.get_dtype(n.output[0])) for n in g.inputs]
    print("--, inputs=", ", ".join(t))
    t = ["{} {}/{}".format(n, g.get_shape(n), g.get_dtype(n)) for n in g.outputs]
    print("--, outputs=", ", ".join(t))
    for node in g.get_nodes():
        input_names = ", ".join(["{} {}/{}".format(n, g.get_shape(n), g.get_dtype(n)) for n in node.input])
        output_names = ", ".join(["{} {}/{}".format(n, g.get_shape(n), g.get_dtype(n)) for n in node.output])
        print("-- {} n={} i={} o={}".format(node.type, node.name, input_names, output_names))
