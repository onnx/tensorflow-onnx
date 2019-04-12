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


@tf_op("ReverseSequence", domain=constants.MICROSOFT_DOMAIN)
class ReverseSequence:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """ReverseSequence"""
        # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
        # T output = ReverseSequence(T input, int32 seqence_lens, @int time_axis, @int batch_axis)
        seq_dim = node.get_attr("seq_dim")
        utils.make_sure(seq_dim is not None, "sequence dim must be given in {}".format(node.name))
        seq_dim = seq_dim.i
        batch_dim = node.get_attr("batch_dim")
        if batch_dim is not None:
            batch_dim = batch_dim.i
        else:
            batch_dim = 0

        output_dtypes = node.output_dtypes
        output_shapes = node.output_shapes
        ctx.remove_node(node.name)
        node = ctx.make_node(
            "ReverseSequence",
            node.input,
            outputs=node.output,
            shapes=output_shapes,
            dtypes=output_dtypes,
            domain=constants.MICROSOFT_DOMAIN,
            attr={"time_axis": seq_dim, "batch_axis": batch_dim}
        )

        seq_len_dtype = ctx.get_dtype(node.input[1])
        utils.make_sure(seq_len_dtype is not None, "dtype of {} is None".format(node.input[1]))
        target_dtype = TensorProto.INT32
        if seq_len_dtype != target_dtype:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=target_dtype)
            ctx.copy_shape(cast_node.input[0], cast_node.output[0])
            ctx.set_dtype(cast_node.output[0], target_dtype)
