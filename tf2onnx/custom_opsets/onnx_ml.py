# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
""" tf2onnx mapping functions for onnx ml domain. """
import logging
from onnx import TensorProto
from tf2onnx import constants
from tf2onnx.handler import tf_op
from tf2onnx import utils

logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring,unnecessary-pass

@tf_op("HashTableV2")
class HashTable:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        """ HashTable will be removed """
        pass


@tf_op("LookupTableFindV2")
class LookupTableFind:
    @classmethod
    def version_8(cls, ctx, node, initialized_tables, **kwargs):
        """ convert lookup to category mapper """
        table_node = node.inputs[0]
        shared_name = table_node.get_attr_value("shared_name")

        utils.make_sure(shared_name in initialized_tables, "Initialized table %s for node %s not found.",
                        shared_name, node.name)

        default_node = node.inputs[2]
        utils.make_sure(default_node.is_const(), "Default value of table lookup must be const.")
        default_val = default_node.get_tensor_value()

        dtype = ctx.get_dtype(node.output[0])
        in_dtype = ctx.get_dtype(node.input[1])
        utils.make_sure(dtype == TensorProto.INT64 and in_dtype == TensorProto.STRING,
                        "Only lookup tables of type string->int64 are currently supported.")

        cats_strings, cats_int64s = initialized_tables[shared_name]
        shape = ctx.get_shape(node.output[0])

        node_name = node.name
        node_inputs = node.input
        node_outputs = node.output
        ctx.remove_node(node.name)
        ctx.make_node("CategoryMapper", domain=constants.AI_ONNX_ML_DOMAIN,
                      name=node_name, inputs=node_inputs[1: 2], outputs=node_outputs,
                      attr={'cats_int64s': cats_int64s, 'cats_strings': cats_strings, 'default_int64': default_val},
                      shapes=[shape], dtypes=[dtype])
        customer_nodes = ctx.find_output_consumers(table_node.output[0])
        if len(customer_nodes) == 0:
            ctx.remove_node(table_node.name)
