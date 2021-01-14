# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Simple tool to guess inputs and outputs of a tensorflow model."""

from __future__ import division
from __future__ import print_function

import argparse
from collections import Counter

import tensorflow as tf

IGNORE_INPUT = ["Const", "ConstV2", "Variable", "VariableV2", "RestoreV2", "Restore"]
IGNORE_OUTPUT = ["NoOp", "Assign", "TensorSummaryV2", "Placeholder"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    return parser.parse_args()


def cleanup_io_name(name):
    """Cleanup op names."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def main():
    args = get_args()

    op_cnt = Counter()
    attr_cnt = Counter()

    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(args.input, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=g):
        inputs = []
        outputs = []
        ops = g.get_operations()
        input_nodes = {}
        shapes = {}
        for node in ops:
            for i in node.inputs:
                input_nodes[i.name] = 1
            for i in node.control_inputs:
                input_nodes[i.name] = 1
            for i in node.outputs:
                try:
                    shape = i.get_shape().as_list()
                    shapes[i.name] = shape
                except:  # pylint: disable=bare-except
                    pass
        for node in ops:
            for i in node.outputs:
                if i.name not in input_nodes:
                    outputs.append(i.name)
            if not node.inputs and not node.control_inputs and node.type not in IGNORE_INPUT:
                if node.outputs:
                    inputs.append(node.outputs[0].name)
            if node.type in ["PlaceHolder"]:
                inputs.append(node.outputs[0].name)
            op_cnt[node.type] += 1
            for a in node.node_def.attr:
                attr_cnt[a] += 1

    print("Ops: {}".format(op_cnt))
    print("Attr: {}".format(attr_cnt))
    print()
    for i in inputs:
        print("input: {} shape={}".format(i, shapes.get(i)))
    for i in outputs:
        print("output: {} shape={}".format(i, shapes.get(i)))


if __name__ == "__main__":
    main()
