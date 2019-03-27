# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
""" tf2onnx mapping functions for ms domain. """

from onnx.onnx_pb import TensorProto
from tf2onnx import constants, utils
from tf2onnx.function.range import make_range_const


# pylint: disable=unused-argument

def range_op1(ctx, node, name, args):
    """Range."""
    # T range = Range(T start, T limit, T delta)
    dtype = node.get_attr_int("Tidx")
    shape = node.output_shapes[0]
    utils.make_sure(dtype is not None, "Tidx of %s is None", node.name)
    ctx.remove_node(node.name)
    make_range(ctx, node.input[0], node.input[1], node.input[2], node.output[0], name, shape, dtype)


def make_range(ctx, start, limit, delta, output, scope_name, shape, dtype):
    if all(ctx.get_node_by_output(n).is_const() for n in [start, limit, delta]) is True:
        make_range_const(ctx, start, limit, delta, output, scope_name, shape, dtype)
    else:
        _make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype)


def _make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype):
    utils.make_sure(
        dtype in [TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64],
        "dtype %s is not supported", dtype)
    ctx.make_node("Range", [start, limit, delta], outputs=[output], name=scope_name, shapes=[shape], dtypes=[dtype],
                  domain=constants.MICROSOFT_DOMAIN)


_OPSET_1 = {
    "Range": (range_op1, []),
}

OPSETS = [
    (1, _OPSET_1),
]
