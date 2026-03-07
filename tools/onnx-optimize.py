# SPDX-License-Identifier: Apache-2.0


"""
A simple tool to try optimizations on onnx graphs.
This makes use of the fact that tensorflow-onnx internal graph representation is onnx
so all graph, rewrite, matching and utility libaries do work which makes things easy.
"""

# pylint: disable=invalid-name,missing-docstring, unused-argument

import argparse
import logging

import onnx
from onnx import helper, shape_inference

from tf2onnx import constants, logging, optimizer
from tf2onnx.graph import GraphUtil
from tf2onnx.late_rewriters import rewrite_channels_first, rewrite_channels_last

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx-optimize")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="onnx input model file")
    parser.add_argument("--output", help="output model file")
    target_options = [constants.TARGET_CHANNELS_LAST, constants.TARGET_CHANNELS_FIRST]
    parser.add_argument("--target", default=",".join(constants.DEFAULT_TARGET), choices=target_options,
                        help="target platform")
    args = parser.parse_args()
    args.target = args.target.split(",")
    return args


def load_graph(fname, target):
    model_proto = onnx.ModelProto()
    with open(fname, "rb") as f:
        data = f.read()
        model_proto.ParseFromString(data)
    g = GraphUtil.create_graph_from_onnx_model(model_proto, target)
    return g, model_proto


def model_shape_inference(onnx_model_proto):
    inferred_model = shape_inference.infer_shapes(onnx_model_proto)
    onnx.checker.check_model(inferred_model)
    return inferred_model


def main():
    args = get_args()

    g, org_model_proto = load_graph(args.input, args.target)

    if g.is_target(constants.TARGET_CHANNELS_FIRST):
        g.reset_nodes(rewrite_channels_first(g, g.get_nodes()))
    if g.is_target(constants.TARGET_CHANNELS_LAST):
        g.reset_nodes(rewrite_channels_last(g, g.get_nodes()))

    g = optimizer.optimize_graph(g)

    onnx_graph = g.make_graph(org_model_proto.graph.doc_string + " (+tf2onnx/onnx-optimize)")

    kwargs = GraphUtil.get_onnx_model_properties(org_model_proto)

    model_proto = helper.make_model(onnx_graph, **kwargs)

    model_proto_inferred = model_shape_inference(model_proto)

    # write onnx graph
    if args.output:
        with open(args.output, "wb") as f:
            f.write(model_proto_inferred.SerializeToString())


if __name__ == "__main__":
    main()
