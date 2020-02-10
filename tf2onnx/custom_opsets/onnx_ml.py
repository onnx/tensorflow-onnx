# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
""" tf2onnx mapping functions for onnx ml domain. """
from tf2onnx import constants
from tf2onnx.handler import tf_op


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
    def version_8(cls, ctx, node, **kwargs):
        """ convert lookup to category mapper """
        table_node = node.inputs[0]
        file_path = table_node.get_attr_value("shared_name")[11:-6]
        cats_int64s = []
        cats_strings = []
        with open(file_path, 'r') as f:
            for i, s in enumerate(f.readlines()):
                cats_int64s.append(i)
                cats_strings.append(s.strip())
        node_name = node.name
        node_inputs = node.input
        node_outputs = node.output
        ctx.remove_node(node.name)
        new_node = ctx.make_node("CategoryMapper", domain=constants.AI_ONNX_ML_DOMAIN,
                                 name=node_name, inputs=node_inputs[1: 2], outputs=node_outputs,
                                 attr={'cats_int64s': cats_int64s, 'cats_strings': cats_strings})
        ctx.set_shape(new_node.name + ":0", [-1])
        customer_nodes = ctx.find_output_consumers(table_node.output[0])
        if len(customer_nodes) == 0:
            ctx.remove_node(table_node.name)
