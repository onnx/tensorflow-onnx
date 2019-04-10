# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx.onnx_opset.logical
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from onnx import onnx_pb
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import common

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("onnx_opset.logical")

# pylint: disable=unused-argument,missing-docstring

def logical_compare_op(ctx, node, **kwargs):
    # T2 output = Greater(T1 x, T1 y), T2=tensor(bool)
    # T2 output = Less(T1 x, T1 y), T2=tensor(bool)
    # Great/Less in opset7 only supports limited types, insert Cast if needed
    if ctx.opset < 9:
        supported_dtypes = [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.DOUBLE
        ]
        target_dtype = onnx_pb.TensorProto.FLOAT
        for inp in node.input:
            if ctx.get_dtype(inp) not in supported_dtypes:
                inp_cast = ctx.insert_new_node_on_input(node, "Cast", inp, to=target_dtype)
                ctx.copy_shape(inp, inp_cast.output[0])
                ctx.set_dtype(inp_cast.output[0], target_dtype)


@tf_op(["LogicalNot", "NotEqual"], onnx_op="Not")
class DirectOp:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        pass


@tf_op(["Equal", "Greater", "Less"])
@tf_op("LogicalAnd", onnx_op="And")
@tf_op("LogicalOr", onnx_op="Or")
class BroadcastOp(common.BroadcastOp):
    pass


@tf_op(["Greater", "Less"])
class Greater:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        common.BroadcastOp.version_4(ctx, node, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        logical_compare_op(ctx, node, **kwargs)


@tf_op("GreaterEqual", onnx_op="Less")
@tf_op("LessEqual", onnx_op="Greater")
class GreaterLessEqual:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        logical_compare_op(ctx, node, **kwargs)
        output_name = node.output[0]
        new_node = ctx.insert_new_node_on_output("Not", output_name, name=utils.make_name(node.name))
        ctx.copy_shape(output_name, new_node.output[0])
        ctx.set_dtype(new_node.output[0], ctx.get_dtype(output_name))
