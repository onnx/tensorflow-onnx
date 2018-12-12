# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.shape_inference - shape inference function for tf2onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
from onnx import onnx_pb


# pylint: disable=logging-not-lazy,missing-docstring,consider-swap-variables


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.shape_inference")


def infer_shape_for_graph(g):
    no_shape_updated = True
    while no_shape_updated:
        no_shape_updated = False
        for o in g.get_nodes():
            updated = infer_shape_for_node(g, o)
            if updated:
                no_shape_updated = True


def infer_shape_for_node(g, node):
    has_unknown_output_shape = False
    for out in node.output:
        out_dtype = g.get_shape(out)
        if out_dtype is None:
            has_unknown_output_shape = True
            break

    if not has_unknown_output_shape:
        return False

    # for those ops, we don't expect all input shapes available to infer output shapes.
    ret = infer_output_shapes_with_partial_inputs(g, node)
    if ret is not None:
        return ret

    # for ops, we need all input shapes ready to infer output shapes.
    are_all_input_shape_ready = True
    no_shape = []
    for i in node.input:
        if g.get_shape(i) is None:
            are_all_input_shape_ready = False
            no_shape.append(i)

    if not are_all_input_shape_ready:
        log.debug("node %s has inputs don't have shape specified, they are: %s", node.name, no_shape)
        return False

    if node.type in ["Cast", "Enter", "Floor", "ReverseSequence", "Sigmoid", "Tanh", "Identity"]:
        return set_shape_from_input(g, node.input[0], node.output[0])

    if node.type in ["Add", "GreaterEqual", "Mul", "RealDiv", "Sub"]:
        return set_shape_from_inputs_broadcast(g, node.input, node.output[0])

    if node.type == "Placeholder":
        # if placeholder shape is not found, try to get it from "shape" attribute.
        shape_attr = node.get_attr("shape")
        new_shape = None
        if shape_attr.type == onnx_pb.TensorProto.INT32:
            new_shape = shape_attr.ints
        elif shape_attr.type == onnx_pb.TensorProto.FLOAT:
            # for scalar placeholder, it's type is float
            val = shape_attr.floats
            if val:
                raise ValueError("placeholder shape has floats value, and not scalar value")
            else:
                new_shape = ()

        if new_shape is not None:
            g.set_shape(node.output[0], new_shape)
            log.debug("set placeholder node [%s] with new shape %s", node.output[0], new_shape)
            return True
        return False

    if node.type == "RandomUniform":
        shape_node = node.inputs[0]
        if not shape_node or shape_node.type != "Shape":
            return False
        return set_shape_from_input(g, shape_node.input[0], node.output[0])

    return False


def infer_output_shapes_with_partial_inputs(g, node):
    if node.type == "Merge":
        s1 = g.get_shape(node.input[0])
        s2 = g.get_shape(node.input[1])
        new_shape = s1
        if s1 is None:
            new_shape = s2

        if new_shape is not None:
            g.set_shape(node.input[0], new_shape)
            g.set_shape(node.input[1], new_shape)
            g.set_shape(node.output[0], new_shape)
            log.debug("set [%s] with new shape %s", node.output[0], new_shape)
            return True
        return False
    if node.type == "Switch":
        new_shape = g.get_shape(node.input[0])
        if new_shape is not None:
            g.set_shape(node.output[0], new_shape)
            g.set_shape(node.output[1], new_shape)
            log.debug("set [%s] with new shape %s", node.output[0], new_shape)
            log.debug("set [%s] with new shape %s", node.output[1], new_shape)
            return True
        return False
    if node.type == "Select":
        new_shape = g.get_shape(node.input[1])
        if new_shape is None:
            new_shape = g.get_shape(node.input[2])
        if new_shape is not None:
            g.set_shape(node.output[0], new_shape)
            g.set_shape(node.input[1], new_shape)
            g.set_shape(node.input[2], new_shape)
            log.debug("set [%s] with new shape %s", node.output[0], new_shape)
            return True
        return False
    return None


def set_shape_from_input(g, input_id, output_id):
    new_shape = g.get_shape(input_id)
    if new_shape is not None:
        g.set_shape(output_id, new_shape)
        log.debug("set [%s] with new shape %s", output_id, new_shape)
        return True
    return False


def set_shape_from_inputs_broadcast(g, input_ids, output_id):
    s1 = g.get_shape(input_ids[0])
    s2 = g.get_shape(input_ids[1])
    new_shape = broadcast_shape_inference(s1, s2)
    if new_shape is not None:
        g.set_shape(output_id, new_shape)
        log.debug("set [%s] with new shape %s", output_id, new_shape)
        return True
    return False


def broadcast_shape_inference(shape_0, shape_1):
    if shape_0 is None:
        return shape_1
    if shape_1 is None:
        return shape_0

    # two dimensions are compatible when they are equal, or one of them is 1
    # compare from last dim
    if len(shape_0) > len(shape_1):
        tmp = shape_0
        shape_0 = shape_1
        shape_1 = tmp

    new_shape = shape_1
    l = len(shape_0)
    if l == 0:
        return new_shape

    i = l - 1
    while i >= 0:
        if shape_0[i] == shape_1[i]:
            # do nothing
            pass
        elif shape_0[i] == 1:
            # do nothing
            pass
        elif shape_1[i] == 1:
            new_shape[i] = shape_0[i]
        # maybe one of them is -1, we can use the other one as real shape.
        elif shape_0[i] == -1:
            pass
        elif shape_1[i] == -1:
            new_shape[i] = shape_0[i]
        else:
            log.warning("two shapes not possible to broadcast, %s, %s", shape_0, shape_1)
            return None
        i -= 1
    return new_shape
