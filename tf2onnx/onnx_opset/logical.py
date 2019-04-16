# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
logical
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from onnx import TensorProto
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import common


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

def _add_cast_to_inputs(graph, node, supported_dtypes, target_dtype):
    is_support = True
    for inp in node.input:
        if graph.get_dtype(inp) not in supported_dtypes:
            is_support = False
            break
    if not is_support:
        for inp in node.input:
            inp_cast = graph.insert_new_node_on_input(node, "Cast", inp, to=target_dtype)
            graph.copy_shape(inp, inp_cast.output[0])
            graph.set_dtype(inp_cast.output[0], target_dtype)


@tf_op(["LogicalNot", "NotEqual"], onnx_op="Not")
class DirectOp:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        pass


@tf_op("LogicalAnd", onnx_op="And")
@tf_op("LogicalOr", onnx_op="Or")
class BroadcastOp(common.BroadcastOp):
    pass


@tf_op("Equal")
class Equal:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        common.BroadcastOp.version_4(ctx, node, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T2 output = Equal(T1, x, T1 y), T1 \in {bool, int32, int64}
        supported_dtypes = [
            TensorProto.BOOL,
            TensorProto.INT32,
            TensorProto.INT64
        ]
        target_dtype = TensorProto.INT32
        _add_cast_to_inputs(ctx, node, supported_dtypes, target_dtype)


@tf_op(["Greater", "Less"])
class GreaterLess:
    @classmethod
    def version_4(cls, ctx, node, **kwargs):
        common.BroadcastOp.version_4(ctx, node, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T2 output = Greater(T1 x, T1 y), T2=tensor(bool)
        # T2 output = Less(T1 x, T1 y), T2=tensor(bool)
        # Great/Less in opset7 only supports limited types, insert Cast if needed
        supported_dtypes = [
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE
        ]
        target_dtype = TensorProto.FLOAT
        _add_cast_to_inputs(ctx, node, supported_dtypes, target_dtype)


@tf_op("GreaterEqual", onnx_op="Less")
@tf_op("LessEqual", onnx_op="Greater")
class GreaterLessEqual:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        GreaterLess.version_7(ctx, node, **kwargs)
        output_name = node.output[0]
        new_node = ctx.insert_new_node_on_output("Not", output_name, name=utils.make_name(node.name))
        ctx.copy_shape(output_name, new_node.output[0])
        ctx.set_dtype(new_node.output[0], ctx.get_dtype(output_name))
