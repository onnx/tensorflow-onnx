# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - select op conversion
"""

from tf2onnx import utils


# pylint: disable=unused-argument,missing-docstring


def select_op7(ctx, node, name, args):
    # T output = Select(bool condition, T x, T y)
    select_x_dtype = ctx.get_dtype(node.input[1])
    select_x_shape = ctx.get_shape(node.input[1])
    cond = node.inputs[0]
    cond_shape = ctx.get_shape(cond.output[0])
    utils.make_sure(select_x_shape is not None and cond_shape is not None, "rank of inputs are needed")

    added_nodes = []
    true_mask = ctx.make_node("Cast", cond.output, attr={"to": select_x_dtype})
    cond_not = ctx.make_node("Not", cond.output)
    false_mask = ctx.make_node("Cast", cond_not.output, attr={"to": select_x_dtype})
    added_nodes.extend([true_mask, cond_not, false_mask])
    # the broadcasting rule of select is different with common rule.
    # for example, shape of input_x is (10), shape of input_y is (10, 2) then common broadcasting rule will fail
    # while in tf "select", input_x will become (10, 1) and repeat at last dimension
    # so reshape node is inserted here
    unsqueeze_dim_num = len(select_x_shape) - len(cond_shape)
    utils.make_sure(unsqueeze_dim_num >= 0, "dim of select_x must not less than cond")
    if unsqueeze_dim_num != 0:
        unsqueeze_dim_start = len(select_x_shape)
        axes = range(unsqueeze_dim_start-1, unsqueeze_dim_start+unsqueeze_dim_num-1)
        true_mask = ctx.make_node("Unsqueeze", true_mask.output, attr={"axes": axes})
        false_mask = ctx.make_node("Unsqueeze", false_mask.output, attr={"axes": axes})
        added_nodes.extend([true_mask, false_mask])

    select_from_true = ctx.make_node("Mul", [true_mask.output[0], node.input[1]])
    select_from_false = ctx.make_node("Mul", [false_mask.output[0], node.input[2]])
    res = ctx.make_node("Add", [select_from_true.output[0], select_from_false.output[0]],
                        name=node.name, outputs=node.output,
                        shapes=[ctx.get_shape(node.output[0])], dtypes=[ctx.get_dtype(node.output[0])])
    return [*added_nodes, select_from_true, select_from_false, res]
