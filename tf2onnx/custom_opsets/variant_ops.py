# SPDX-License-Identifier: Apache-2.0

""" tf2onnx mapping functions for string ops using contrib ops domain. """
import logging
from onnx.onnx_pb import TensorProto
from tf2onnx import constants
from tf2onnx.handler import tf_op
from tf2onnx import utils

logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring

@tf_op("RaggedTensorToTensor", domain=constants.CONTRIB_OPS_DOMAIN)
class RaggedTensorToTensorOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.CONTRIB_OPS_DOMAIN
        node.type = "RaggedTensorToDense"
        if ctx.get_dtype(node.input[1]) == TensorProto.STRING:
            node.type = "StringRaggedTensorToDense"
        if len(node.input) == 5:
            # 5 inputs: shape, values, default_value, row_partition_tensors, row_partition_types
            # 5 inputs from 0 to 4.
            # The ONNX version merges only uses the data (input 1) and the indices (input 3)
            ctx.remove_input(node, node.input[4])
            if len(node.input) == 5:
                del node.input[4]
        utils.make_sure(len(node.input) == 4,
                        "[RaggedTensorToTensorOp] the node should have 4 inputs not %r.",
                        len(node.input))
