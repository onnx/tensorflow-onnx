# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.shape_inference - shape inference function for tf2onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import os
import tensorflow as tf
from distutils.version import LooseVersion
from tf2onnx import utils

# pylint: disable=logging-not-lazy,missing-docstring,consider-swap-variables


logger = logging.getLogger(__name__)


def tf_cpp_infer_shape(tf_graph):
    """Invoke tensorflow shape inference by re-importing graph_def."""
    # invoke c api if tf version is below 1.8
    if utils.get_tf_version() < LooseVersion("1.8"):
        os.environ["TF_C_API_GRAPH_CONSTRUCTION"] = "1"

    graph_def = tf_graph.as_graph_def(add_shapes=True)
    with tf.Graph().as_default() as inferred_graph:
        tf.import_graph_def(graph_def, name="")
    return inferred_graph


def infer_shape_for_graph(tf_graph):
    """
    Infer shape for Tensorflow ops.
    Tensorflow explicitly sets shape for some ops in python code, such as Switch, Merge and TensorArrayGather.
    These shapes may be lost after freezing TF graph to graph_def without add_shapes=True.
    To bring these shapes back, we implement our own shape inference for these control flow ops based on one assumption:
    **outputs of Merge op have the same shape (at least the same rank) of its inputs**.
    With this assumption, our shape inference can handle:
        1. in tf.cond, outputs of two branches have the same rank.
        2. in tf.while_loop, loop variables don't change their rank.
    """
    no_shape_updated = True
    while no_shape_updated:
        no_shape_updated = False
        for o in tf_graph.get_operations():
            updated = infer_shape_for_node(o)
            if updated:
                no_shape_updated = True
        tf_graph = tf_cpp_infer_shape(tf_graph)
    return tf_graph


def infer_shape_for_node(op):
    has_unknown_output_shape = any(utils.get_shape_from_tf_output(out) is None for out in op.outputs)

    if not has_unknown_output_shape:
        return False

    # for those ops, we don't expect all input shapes available to infer output shapes.
    ret = infer_output_shapes_with_partial_inputs(op)
    if ret is not None:
        return ret

    if op.type == "Placeholder":
        # if placeholder shape is not found, try to get it from "shape" attribute.
        attr_shape = utils.get_tf_shape(op)
        if attr_shape is not None:
            new_shape = list(attr_shape)
            op.outputs[0].set_shape(new_shape)
            logger.debug("set placeholder node [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        logger.warning("Shape of placeholder %s is unknown, treated it as a scalar", op.name)
        op.outputs[0].set_shape([])
        return True

    return False


def infer_output_shapes_with_partial_inputs(op):
    if op.type == "Merge":
        s1 = utils.get_shape_from_tf_output(op.inputs[0])
        s2 = utils.get_shape_from_tf_output(op.inputs[1])
        new_shape = None
        if s1 is None and s2 is None:
            return False
        if s1 is None and s2 is not None:
            new_shape = s2
        if s1 is not None and s2 is None:
            new_shape = s1

        if new_shape is not None:
            op.inputs[0].set_shape(new_shape)
            op.inputs[1].set_shape(new_shape)
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True

        # inputs' shapes both exist
        if s1 != s2:
            if len(s1) != len(s2):
                logger.warning("Shapes of Merge %s have different ranks: %s, %s", op.name, len(s1), len(s2))
                return False

            logger.warning("Inputs of Merge %s have different shapes: %s, %s, but the same rank", op.name, s1, s2)
            new_shape = _merge_shapes_for_tf(s1, s2)
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
        else:
            new_shape = s1
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)

        return True

    if op.type == "Switch":
        new_shape = utils.get_shape_from_tf_output(op.inputs[0])
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            op.outputs[1].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[1].name, new_shape)
            return True
        return False

    if op.type == "Enter":
        new_shape = utils.get_shape_from_tf_output(op.inputs[0])
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    if op.type == "TensorArrayGatherV3":
        # TensorArrayGatherV3's output: all of the elem in the TensorArray,
        # concatenated along a new axis (the new dimension 0), so shape of TensorArray should be found first.
        # And TensorArrayWrite will write elem to TensorArray, so shape of TensorArray can be got from TensorArrayWrite
        # so the process is: first find TensorArrayWrite and then get TensorArray's shape,
        # and finally add one dim to the shape is shape of TensorArrayGather

        handle_op = op.inputs[0].op
        if handle_op.type != "TensorArrayV3":
            return False

        # find TensorArrayWrite
        tensor_array_write_op = _find_tensorarray_write(handle_op)
        if not tensor_array_write_op:
            return False
        # get TensorArray shape from input tensor of the found TensorArrayWrite node
        # value_op = tensor_array_write_op.inputs[2].op
        shape = utils.get_shape_from_tf_output(tensor_array_write_op.inputs[2])
        # update TensorArray's shape info
        if shape is not None:
            new_shape = [None] + shape
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    if op.type == "TensorArrayReadV3":
        # Read an element from the TensorArray into output value.
        flow_in_op = op.inputs[2].op
        if flow_in_op.type != "Enter":
            return False

        scatter_op = flow_in_op.inputs[0].op
        if scatter_op.type != "TensorArrayScatterV3":
            return False

        value_shape_before_scatter = utils.get_shape_from_tf_output(scatter_op.inputs[2])
        if value_shape_before_scatter is None:
            return False

        new_shape = value_shape_before_scatter[1:]
        if new_shape is not None:
            op.outputs[0].set_shape(new_shape)
            logger.debug("set [%s] with new shape %s", op.outputs[0].name, new_shape)
            return True
        return False

    return None


def _find_tensorarray_write(op):
    utils.make_sure(op.type == "TensorArrayV3", "node should be tensorarray")

    tensor_array_consumers = op.outputs[0].consumers()
    for i in tensor_array_consumers:
        if i.type == "Enter":
            consumer_ops = i.outputs[0].consumers()
            for j in consumer_ops:
                if j.type == "TensorArrayWriteV3":
                    return j
    return None


def _merge_shapes_for_tf(shape1, shape2):
    """
    Merge 2 shapes, return merged shape, set unknown for dims with different values.
    Raise exception for mismatch.
    """
    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1

    utils.make_sure(utils.is_list_or_tuple(shape1), "invalid type for shape1")
    utils.make_sure(utils.is_list_or_tuple(shape2), "invalid type for shape2")
    utils.make_sure(len(shape1) == len(shape2), "shapes rank mismatch: shape1=%s, shape2=%s", shape1, shape2)

    merged = []
    for d1, d2 in zip(shape1, shape2):
        d = d1
        if d1 is None:
            d = d2
        elif not d2 is None:
            # None means unknown in tensorflow
            d = None
        merged.append(d)
    return merged
