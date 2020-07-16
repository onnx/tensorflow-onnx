# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tensor
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

import numpy as np
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto

from tf2onnx import constants, utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.handler import tf_op
from tf2onnx.onnx_opset import nn, math

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


@tf_op("FakeQuantWithMinMaxArgs")
class FakeQuantWithMinMaxArgs:
    # see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fake-quant-with-min-max-args
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # hack to make up for the missing onnx pack op
        amin = node.get_attr("min").f
        amax = node.get_attr("max").f
        narrow_range = node.get_attr("narrow_range").i
        num_bits = node.get_attr("num_bits").i
        
        if narrow_range:
            raise RuntimeError(
                "Unable to convert node FakeQuantWithMinMaxArgs with "
                "narrow_range=%r" % narrow_range)
        
        if 0 < amin < amax:
            min_adj = 0
            max_adj = amax - amin
            scale = 1.
        elif amin < amax < 0:
            min_adj = amin - amax
            max_adj = 0
            scale = 1.
        elif amin <= 0 <= amax:
            scale = (amax - amin) / (2 ** num_bits - 1)
            min_adj = scale * int(amin / scale)
            max_adj = amax + min_adj - amin
        else:
            raise RuntimeError(
                "Unable to convert node FakeQuantWithMinMaxArgs with "
                "min=%f and max=%f" % (amin, amax))

        dtype = ctx.get_dtype(node.input[0])
        shape = ctx.get_shape(node.input[0])

        new_node = ctx.make_node(
            "QuantizeLinear", [node.input[0], pb_scale, y_zero_point],
            op_name_scope=node.name, attr={"axes": [axis]},
            shapes=[shape], dtypes=[idtype])
        output_name = new_node.output[0]
        node.input[i] = output_name

        ctx.remove_node(node.name)

        last_node = ctx.make_node(
            "DequantizeLinear", [new_node.output[0], x_scale, x_zero_point],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[dtype])
        ctx.replace_all_inputs(ctx.get_nodes(), node.output[0], last_node.output[0])


