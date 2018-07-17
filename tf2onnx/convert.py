# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
python -m tf2onnx.convert : tool to convert a frozen tensorflow to onnx
"""

from __future__ import division
from __future__ import print_function

import argparse
import re
import sys

import onnx
import tensorflow as tf
import tf2onnx.utils
from tf2onnx.tfonnx import process_tf_graph, tf_optimize, DEFAULT_TARGET, POSSIBLE_TARGETS
from onnx import helper


_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model file")
    parser.add_argument("--output", help="output model file")
    parser.add_argument("--inputs", required=True, help="model input_names")
    parser.add_argument("--outputs", required=True, help="model output_names")
    parser.add_argument("--opset", type=int, default=None, help="highest opset to use")
    parser.add_argument("--custom-ops", help="list of custom ops")
    parser.add_argument("--unknown-dim", type=int, default=1, help="default for unknown dimensions")
    parser.add_argument("--target", default=",".join(DEFAULT_TARGET), help="target platform")
    parser.add_argument("--continue_on_error", help="continue_on_error", action="store_true")
    parser.add_argument("--verbose", help="verbose output", action="store_true")
    args = parser.parse_args()

    args.shape_override = None
    if args.inputs:
        inputs = []
        shapes = {}
        # input takes in most cases the format name:0, where 0 is the output number
        # in some cases placeholders don't have a rank which onnx can't handle so we let uses override the shape
        # by appending the same, ie : [1,28,28,3]
        #
        pattern = r"(?:([\w:]+)(\[[\d,]+\])?),?"
        splits = re.split(pattern, args.inputs)
        for i in range(1, len(splits), 3):
            inputs.append(splits[i])
            if splits[i+1] is not None:
                shapes[splits[i]] = [int(n) for n in splits[i+1][1:-1].split(",")]
        args.inputs = inputs
        args.shape_override = shapes
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.target:
        args.target = args.target.split(",")
        for target in args.target:
            if target not in POSSIBLE_TARGETS:
                print("unknown target ", target)
                sys.exit(1)

    return args


def default_custom_op_handler(ctx, node, name, args):
    node.domain = _TENSORFLOW_DOMAIN
    return node


def main():
    args = get_args()

    print("using tensorflow={}, onnx={}".format(tf.__version__, onnx.__version__))

    # override unknown dimensions from -1 to 1 (aka batchsize 1) since not every runtime does
    # support unknown dimensions.
    tf2onnx.utils.ONNX_UNKNOWN_DIMENSION = args.unknown_dim

    if args.custom_ops:
        # default custom ops for tensorflow-onnx are in the "tf" namespace
        custom_ops = {op: default_custom_op_handler for op in args.custom_ops.split(",")}
        extra_opset = [helper.make_opsetid(_TENSORFLOW_DOMAIN, 1)]
    else:
        custom_ops = {}
        extra_opset = None

    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(args.input, 'rb') as f:
        graph_def.ParseFromString(f.read())
    graph_def = tf_optimize(None, args.inputs, args.outputs, graph_def)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph) as sess:
        g = process_tf_graph(tf_graph,
                             continue_on_error=args.continue_on_error,
                             verbose=args.verbose,
                             target=args.target,
                             opset=args.opset,
                             custom_op_handlers=custom_ops,
                             extra_opset=extra_opset,
                             shape_override=args.shape_override)

    model_proto = g.make_model(
        "converted from {}".format(args.input), args.inputs, args.outputs,
        optimize=not args.continue_on_error)

    # write onnx graph
    if args.output:
        with open(args.output, "wb") as f:
            f.write(model_proto.SerializeToString())


main()
