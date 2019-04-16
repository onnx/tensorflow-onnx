# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
quantitize_weights.py - simple script to quantitize weights (not the model) to 8 bits.
"""

from __future__ import division
from __future__ import print_function

import argparse
import logging
import numpy as np
from onnx import ModelProto, helper, onnx_pb, numpy_helper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantitize_weights")


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    parser.add_argument("--verbose", help="verbose", action="store_true")
    args = parser.parse_args()
    return args


def eight_bit_dequantitize(w_in, zp, scale):
    w = w_in * scale + zp
    w = w.astype("float32")
    return w


def eight_bit_quantitize(w_in):
    """quantitize to 8 bit as scale and zeropoint"""
    low = np.min(w_in)
    high = np.max(w_in)
    scale = (high - low) / 256.
    w = (w_in - low) / scale
    w_out = w.astype("uint8")
    return w_out, low, scale


def _port_name(name):
    return name + "__out"


def _make_node(nodes, op, name, inputs, **kwargs):
    node = helper.make_node(op, inputs, [_port_name(name)], name=name, **kwargs)
    nodes.append(node)


def _compose_quantitize(nodes, weights, zp, scale, name):
    """
    Compose Dequantitize(input, zeropoint, scale).
    """
    name_zp = name + "_zeropoint"
    name_scale = name + "_scale"
    name_add = name + "_add"
    name_mul = name + "_mul"
    name_cast = name + "_cast"

    # add zeropoint and scale as initializers
    weights.append(numpy_helper.from_array(np.array(zp, dtype=np.float32), name_zp))
    weights.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), name_scale))

    # insert ops to dequantitize
    _make_node(nodes, "Cast", name_cast, [name], to=onnx_pb.TensorProto.FLOAT)
    _make_node(nodes, "Mul", name_mul, [_port_name(name_cast), name_scale])
    _make_node(nodes, "Add", name_add, [_port_name(name_mul), name_zp])

    return _port_name(name_add)


def stats(a):
    return {"mean": a.mean(), "std": a.std(), "max": a.max(), "min": a.min()}


def quantitize_graph(g, verbose=False):
    """Quantitize graph."""
    new_weights = []
    quantitized_weights = []
    nodes = []
    remap = {}
    remove = []

    for i, w in enumerate(g.initializer):
        # only quantitize float32
        if w.data_type != onnx_pb.TensorProto.FLOAT:
            continue
        w_np = numpy_helper.to_array(w)
        # only look at sizes >= 32 elements
        if w_np.size < 32:
            continue

        # weights we want to quantitize
        remove.append(i)
        name = w.name
        if verbose:
            logger.info("quantitizing %s", name)
        w_quant, zp, scale = eight_bit_quantitize(w_np)
        nw = numpy_helper.from_array(w_quant, name=name)
        if verbose:
            w_dequant = eight_bit_dequantitize(w_quant, zp, scale)
            rtol = np.abs(w_dequant - w_np)
            s = {}
            for j in [1.0, 5.0, 10.0, 20.0]:
                above_rtol = np.sum(rtol > np.abs(j * w_np / 100.)) / w_np.size
                s["> " + str(j) + "%"] = "{:.2f}".format(100. * above_rtol)
            logger.info("above_rtol: %s", str(s))
            logger.info("raw:   %s", stats(w_np))
            logger.info("quant: %s", stats(w_dequant))
        output_name = _compose_quantitize(nodes, new_weights, zp, scale, name)
        remap[name] = output_name
        quantitized_weights.append(nw)

    # few things to do to initializers and graph inputs:

    # 1. remove initializers that got quantitized
    for i in reversed(remove):
        del g.initializer[i]

    # 2. add quantitized to initializers
    g.initializer.extend(new_weights)
    g.initializer.extend(quantitized_weights)

    # 3. modify the type of weights that we quantitized
    modified = {w.name: w for w in quantitized_weights}
    new_inputs = []
    remove = []
    for i, inp in enumerate(g.input):
        w = modified.get(inp.name)
        if w is not None:
            new_inputs.append(helper.make_tensor_value_info(w.name, w.data_type, w.dims))
            remove.append(i)
    for i in reversed(remove):
        del g.input[i]

    # 4. add new weights as inputs
    for w in new_weights:
        tv = helper.make_tensor_value_info(w.name, w.data_type, w.dims)
        new_inputs.append(tv)
    g.input.extend(new_inputs)

    # 5. rewrite consumers of the quantitized weights
    for node in g.node:
        for i, name in enumerate(node.input):
            new_name = remap.get(name)
            if new_name is not None:
                node.input[i] = new_name

    # 6. add composed nodes to graph, new nodes in the front
    nodes.extend(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return g


def main():
    args = _get_args()

    # read onnx graph
    with open(args.input, "rb") as f:
        data = f.read()
        model_proto = ModelProto()
        model_proto.ParseFromString(data)

    # quantitize weights
    g = quantitize_graph(model_proto.graph, args.verbose)

    # write quantitized graph
    with open(args.output, "wb") as f:
        # create model proto
        model_proto_out = helper.make_model(g,
                                            producer_name="quantized {}".format(model_proto.producer_name),
                                            producer_version=model_proto.producer_version,
                                            opset_imports=model_proto.opset_import)
        f.write(model_proto_out.SerializeToString())


if __name__ == "__main__":
    main()
