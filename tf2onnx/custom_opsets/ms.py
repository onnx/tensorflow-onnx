# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
""" tf2onnx mapping functions for ms domain. """

from onnx.onnx_pb import TensorProto
from tf2onnx import constants, utils
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import controlflow


# pylint: disable=unused-argument,missing-docstring

def make_range(ctx, start, limit, delta, output, scope_name, shape, dtype):
    if all(ctx.get_node_by_output(n).is_const() for n in [start, limit, delta]) is True:
        controlflow.make_range_const(ctx, start, limit, delta, output, scope_name, shape, dtype)
    else:
        _make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype)


def _make_range_non_const(ctx, start, limit, delta, output, scope_name, shape, dtype):
    utils.make_sure(
        dtype in [TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64],
        "dtype %s is not supported", dtype)
    ctx.make_node("Range", [start, limit, delta], outputs=[output], name=scope_name, shapes=[shape], dtypes=[dtype],
                  domain=constants.MICROSOFT_DOMAIN)


@tf_op("Range", domain=constants.MICROSOFT_DOMAIN)
class Range:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """Range."""
        # T range = Range(T start, T limit, T delta)
        dtype = node.get_attr_int("Tidx")
        shape = node.output_shapes[0]
        utils.make_sure(dtype is not None, "Tidx of %s is None", node.name)
        ctx.remove_node(node.name)
        make_range(ctx, node.input[0], node.input[1], node.input[2], node.output[0], node.name, shape, dtype)


@tf_op("Cumsum")
class CumSum:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        attrs = {}
        exclusive = node.get_attr('exclusive')
        if exclusive:
            attrs['exclusive'] = exclusive.i
        reverse = node.get_attr('reverse')
        if reverse:
            attrs['reverse'] = reverse.i
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node("CumSum", inputs=node.input, outputs=node.output, name=node.name,
                      shapes=shapes, dtypes=dtypes,
                      domain=constants.MICROSOFT_DOMAIN, attr=attrs)


@tf_op("Round")
class Round:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node("Round", inputs=node.input, outputs=node.output, name=node.name,
                      shapes=shapes, dtypes=dtypes,
                      domain=constants.MICROSOFT_DOMAIN)
