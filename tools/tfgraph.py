# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
from collections import Counter

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

IGNORE_INPUT = ["Const", "ConstV2", "Variable", "VariableV2", "RestoreV2", "Restore"]
IGNORE_OUTPUT = ["NoOp", "Assign", "TensorSummaryV2", "Placeholder"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    return parser.parse_args()


def tf_optimize(sess, input_names, output_names, graph_def):
    transforms = [
        "remove_nodes(op=Identity, op=CheckNumerics)",
        "fold_batch_norms",
        "fold_old_batch_norms"
        # fails: "fold_constants(ignore_errors=true)",
    ]
    needed_names = input_names + output_names
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)
    graph_def = TransformGraph(graph_def, input_names, output_names, transforms)
    return graph_def


def cleanup_io_name(name):
    # name = name.replace("\^", "")
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
    with tf.Session(graph=g) as sess:
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
                except:
                    pass
        for node in ops:
            name = node.name
            for i in node.outputs:
                if i.name not in input_nodes:
                    outputs.append(i.name)
            if len(node.inputs) == 0 and len(node.control_inputs) == 0 and node.type not in IGNORE_INPUT:
                if len(node.outputs) > 0:
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
