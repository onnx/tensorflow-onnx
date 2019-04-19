# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
python -m tf2onnx.convert : tool to convert a frozen tensorflow graph to onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import tensorflow as tf

from tf2onnx.graph import GraphUtil
from tf2onnx.tfonnx import process_tf_graph, tf_optimize
from tf2onnx import constants, loader, logging, utils


# pylint: disable=unused-argument


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input from graphdef")
    parser.add_argument("--graphdef", help="input from graphdef")
    parser.add_argument("--saved-model", help="input from saved model")
    parser.add_argument("--checkpoint", help="input from checkpoint")
    parser.add_argument("--output", help="output model file")
    parser.add_argument("--inputs", help="model input_names")
    parser.add_argument("--outputs", help="model output_names")
    parser.add_argument("--opset", type=int, default=None, help="opset version to use for onnx domain")
    parser.add_argument("--custom-ops", help="list of custom ops")
    parser.add_argument("--extra_opset", default=None,
                        help="extra opset with format like domain:version, e.g. com.microsoft:1")
    parser.add_argument("--target", default=",".join(constants.DEFAULT_TARGET), choices=constants.POSSIBLE_TARGETS,
                        help="target platform")
    parser.add_argument("--continue_on_error", help="continue_on_error", action="store_true")
    parser.add_argument("--verbose", "-v", help="verbose output, option is additive", action="count")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument("--fold_const", help="enable tf constant_folding transformation before conversion",
                        action="store_true")
    # experimental
    parser.add_argument("--inputs-as-nchw", help="transpose inputs as from nhwc to nchw")
    # depreciated, going to be removed some time in the future
    parser.add_argument("--unknown-dim", type=int, default=-1, help="default for unknown dimensions")
    args = parser.parse_args()

    args.shape_override = None
    if args.input:
        # for backward compativility
        args.graphdef = args.input
    if args.graphdef or args.checkpoint:
        if not args.input and not args.outputs:
            raise ValueError("graphdef and checkpoint models need to provide inputs and outputs")
    if not any([args.graphdef, args.checkpoint, args.saved_model]):
        raise ValueError("need input as graphdef, checkpoint or saved_model")
    if args.inputs:
        args.inputs, args.shape_override = utils.split_nodename_and_shape(args.inputs)
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.inputs_as_nchw:
        args.inputs_as_nchw = args.inputs_as_nchw.split(",")
    if args.target:
        args.target = args.target.split(",")

    if args.extra_opset:
        tokens = args.extra_opset.split(':')
        if len(tokens) != 2:
            raise ValueError("invalid extra_opset argument")
        args.extra_opset = [utils.make_opsetid(tokens[0], int(tokens[1]))]

    return args


def default_custom_op_handler(ctx, node, name, args):
    node.domain = constants.TENSORFLOW_OPSET.domain
    return node


def main():
    args = get_args()
    logging.basicConfig(level=logging.get_verbosity_level(args.verbose))
    if args.debug:
        utils.set_debug_mode(True)

    logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)

    # override unknown dimensions from -1 to 1 (aka batchsize 1) since not every runtime does
    # support unknown dimensions.
    utils.ONNX_UNKNOWN_DIMENSION = args.unknown_dim

    extra_opset = args.extra_opset or []
    custom_ops = {}
    if args.custom_ops:
        # default custom ops for tensorflow-onnx are in the "tf" namespace
        custom_ops = {op: (default_custom_op_handler, []) for op in args.custom_ops.split(",")}
        extra_opset.append(constants.TENSORFLOW_OPSET)

    # get the frozen tensorflow model from graphdef, checkpoint or saved_model.
    if args.graphdef:
        graph_def, inputs, outputs = loader.from_graphdef(args.graphdef, args.inputs, args.outputs)
        model_path = args.graphdef
    if args.checkpoint:
        graph_def, inputs, outputs = loader.from_checkpoint(args.checkpoint, args.inputs, args.outputs)
        model_path = args.checkpoint
    if args.saved_model:
        graph_def, inputs, outputs = loader.from_saved_model(args.saved_model, args.inputs, args.outputs)
        model_path = args.saved_model

    # todo: consider to enable const folding by default?
    graph_def = tf_optimize(inputs, outputs, graph_def, args.fold_const)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(tf_graph,
                             continue_on_error=args.continue_on_error,
                             target=args.target,
                             opset=args.opset,
                             custom_op_handlers=custom_ops,
                             extra_opset=extra_opset,
                             shape_override=args.shape_override,
                             input_names=inputs,
                             output_names=outputs,
                             inputs_as_nchw=args.inputs_as_nchw)

    model_proto = g.make_model("converted from {}".format(model_path))

    logger.info("")
    model_proto = GraphUtil.optimize_model_proto(model_proto)

    # write onnx graph
    logger.info("")
    logger.info("Successfully converted TensorFlow model %s to ONNX", model_path)
    if args.output:
        utils.save_protobuf(args.output, model_proto)
        logger.info("ONNX model is saved at %s", args.output)
    else:
        logger.info("To export ONNX model to file, please run with `--output` option")


if __name__ == "__main__":
    main()
