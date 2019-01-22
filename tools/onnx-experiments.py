# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
A simple tool to try optimizations on onnx graphs.
This makes use of the fact that tensorflow-onnx internal graph representation is onnx
so all graph, rewrite, matching and utility libaries do work which makes things easy.
"""

# pylint: disable=invalid-name,missing-docstring, unused-argument

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import traceback

import numpy as np
import onnx
from onnx import numpy_helper

import tf2onnx.utils
from tf2onnx.graph import Graph

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("onnx-experiments")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="onnx input model file")
    parser.add_argument("--output", help="output model file")
    args = parser.parse_args()
    return args


def load_graph(fname):
    with open(fname, "rb") as f:
        data = f.read()
        model_proto = onnx.ModelProto()
        model_proto.ParseFromString(data)
        onnx_nodes = model_proto.graph.node
        output_names = []

        # some pytorch model had empty names - make one up
        for node in onnx_nodes:
            if not node.name:
                node.name = tf2onnx.utils.make_name("was_empty")

        g = Graph(onnx_nodes, output_shapes={}, dtypes={}, output_names=output_names)
        for i in model_proto.graph.initializer:
            v = numpy_helper.to_array(i)
            name = i.name
            g.initializers[name] = i
            dtype = i.data_type
            g.set_dtype(name, dtype)
            g.set_shape(name, v.shape)
        for i in model_proto.graph.input:
            name = i.name
            if name in g.initializers:
                # ignore if it is not a model input
                continue
            shape = [j.dim_value if hasattr(i.type.tensor_type, "dim_value") else -1
                     for j in i.type.tensor_type.shape.dim]
            dtype = i.type.tensor_type.elem_type
            g.set_dtype(name, dtype)
            g.set_shape(name, shape)
            g.add_graph_input(name, dtype, shape)
        for i in model_proto.graph.output:
            name = i.name
            shape = [j.dim_value if hasattr(i.type.tensor_type, "dim_value") else -1
                     for j in i.type.tensor_type.shape.dim]
            dtype = i.type.tensor_type.elem_type
            g.set_dtype(name, dtype)
            g.set_shape(name, shape)
            output_names.append(name)

        # TODO: this is a hack in case a output name does not follow tensorflow convention
        for node in g.get_nodes():
            for name in node.output:
                g._nodes_by_name[name] = node  # pylint: disable=protected-access

    return g, model_proto.producer_name


def sample_rewrite(g, ops):
    return ops


def rewrite_constant_fold(g, ops):

    func_map = {
        "Transpose": not None,
        "Unsqueeze": not None,
        "Slice": not None,
        "Add": np.add,
        "Cast": np.cast,
        "Mul": np.multiply,
        "Sqrt": np.sqrt,
        "Sub": np.subtract,
    }

    # pylint: disable=too-many-nested-blocks
    keep_looking = True
    while keep_looking:
        keep_looking = False
        for idx, op in enumerate(ops):
            if op.is_deleted():
                continue

            inputs = []
            for node in op.inputs:
                if node and node.is_const():
                    inputs.append(node.get_tensor_value(as_list=False))

            if inputs and len(op.input) == len(inputs):
                func = func_map.get(op.type)
                if func is None:
                    log.info("can fold but don't know how, type=%s, name=%s", op.type, op.name)
                    continue
                try:
                    log.info("folding node type=%s, name=%s", op.type, op.name)
                    if op.type == "Cast":
                        dst = op.get_attr_int("to")
                        np_type = dst
                        val = np.cast[np_type](*inputs)
                    elif op.type == "Transpose":
                        perm = op.get_attr("perm").ints
                        val = np.transpose(inputs[0], perm)
                    elif op.type == "Unsqueeze":
                        axis = op.get_attr_int("axis")
                        val = np.expand_dims(inputs[0], axis=axis)
                    elif op.type == "Slice":
                        axis = op.get_attr_int("axis")
                        if axis != 0:
                            log.info("can fold slice with axis!=0, type=%s, name=%s", op.type, op.name)
                            continue
                        starts = op.get_attr_int("starts")
                        ends = op.get_attr_int("ends")
                        if starts == 0 and ends == 0:
                            val = inputs[0][starts:ends]
                        else:
                            val = inputs[0]
                    else:
                        val = func(*inputs)

                    new_node_name = tf2onnx.utils.make_name(op.name)
                    new_output_name = new_node_name
                    old_output_name = op.output[0]
                    old_node_name = op.name
                    log.debug("create const node [%s] replacing [%s]", new_node_name, old_node_name)
                    ops[idx] = g.make_const(new_node_name, val)
                    consumers = g.find_output_consumers(old_output_name)
                    if consumers:
                        for consumer in consumers:
                            g.replace_input(consumer, old_output_name, new_output_name)
                    for node in op.inputs:
                        node.set_deleted()
                    keep_looking = True
                except Exception as ex:  # pylint: disable=broad-except
                    tb = traceback.format_exc()
                    log.info("exception: %s, details: %s", ex, tb)
                    # pylint: enable=too-many-nested-blocks
    return g.remove_deleted_nodes(ops)


def main():
    args = get_args()

    g, producer_name = load_graph(args.input)

    rewriters = [sample_rewrite,
                 rewrite_constant_fold]
    ops = g.get_nodes()
    stats = g.dump_node_statistics()

    for rewrite in rewriters:
        ops = rewrite(g, ops)

    try:
        g.topological_sort(ops)
    except Exception as ex:  # pylint: disable=broad-except,unused-variable
        log.error("graph has cycles, ignored ...")

    model_proto = g.make_model(producer_name)

    print("before:", stats)
    stats.subtract(g.dump_node_statistics())
    print("removed:", stats)

    # write onnx graph
    if args.output:
        with open(args.output, "wb") as f:
            f.write(model_proto.SerializeToString())


if __name__ == "__main__":
    main()
