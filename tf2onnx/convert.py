# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
python -m tf2onnx.convert : tool to convert a frozen tensorflow graph to onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import onnx
from onnx import helper
import tensorflow as tf

import tf2onnx.utils
from tf2onnx.optimizer.transpose_optimizer import TransposeOptimizer
from tf2onnx.tfonnx import process_tf_graph, tf_optimize, DEFAULT_TARGET, POSSIBLE_TARGETS

_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"

# pylint: disable=unused-argument


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model file")
    parser.add_argument("--output", help="output model file")
    parser.add_argument("--inputs", required=True, help="model input_names")
    parser.add_argument("--outputs", required=True, help="model output_names")
    parser.add_argument("--opset", type=int, default=None, help="highest opset to use")
    parser.add_argument("--custom-ops", help="list of custom ops")
    parser.add_argument("--target", default=",".join(DEFAULT_TARGET), choices=POSSIBLE_TARGETS, help="target platform")
    parser.add_argument("--continue_on_error", help="continue_on_error", action="store_true")
    parser.add_argument("--verbose", help="verbose output", action="store_true")
    parser.add_argument("--fold_const", help="enable tf constant_folding transformation before conversion",
                        action="store_true")
    # experimental
    parser.add_argument("--inputs-as-nchw", help="transpose inputs as from nhwc to nchw")
    # depreciated, going to be removed some time in the future
    parser.add_argument("--unknown-dim", type=int, default=-1, help="default for unknown dimensions")
    args = parser.parse_args()

    args.shape_override = None
    if args.inputs:
        args.inputs, args.shape_override = tf2onnx.utils.split_nodename_and_shape(args.inputs)
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.inputs_as_nchw:
        args.inputs_as_nchw = args.inputs_as_nchw.split(",")
    if args.target:
        args.target = args.target.split(",")

    return args


def default_custom_op_handler(ctx, node, name, args):
    node.domain = _TENSORFLOW_DOMAIN
    return node


def main():
    args = get_args()

    opset = tf2onnx.utils.find_opset(args.opset)
    print("using tensorflow={}, onnx={}, opset={}, tfonnx={}/{}".format(
        tf.__version__, onnx.__version__, opset,
        tf2onnx.__version__, tf2onnx.version.git_version[:6]))

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
    with tf.gfile.GFile(args.input, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # todo: consider to enable const folding by default?
    graph_def = tf_optimize(args.inputs, args.outputs, graph_def, args.fold_const)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(tf_graph,
                             continue_on_error=args.continue_on_error,
                             verbose=args.verbose,
                             target=args.target,
                             opset=args.opset,
                             custom_op_handlers=custom_ops,
                             extra_opset=extra_opset,
                             shape_override=args.shape_override,
                             output_names=args.outputs,
                             inputs_as_nchw=args.inputs_as_nchw)

    optimizer = TransposeOptimizer(g, args.outputs, args.verbose is not None)
    optimizer.optimize()

    model_proto = g.make_model(
        "converted from {}".format(args.input),
        optimize=not args.continue_on_error)

    # write onnx graph
    if args.output:
        with open(args.output, "wb") as f:
            f.write(model_proto.SerializeToString())


if __name__ == "__main__":
    main()
