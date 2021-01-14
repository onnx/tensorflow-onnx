# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
generator
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import onnx_pb, numpy_helper
from tf2onnx import utils
from tf2onnx.handler import tf_op

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

@tf_op(["Const", "ConstV2"])
class DirectOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass


@tf_op(["RandomNormal", "RandomUniform"])
class RandomOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # in tf-2.0 grappler optimizes the graph pretty well and our matching logic
        # in the rewriter does not trigger. grappler will send the random uniform
        # with shape as input so we need to pickup the input here and if the shape is
        # const we make it an attribute.
        seed = node.get_attr("seed")
        node.set_attr("seed", float(seed.f))
        if len(node.input) > 0 and node.inputs[0].is_const():
            shape = node.inputs[0].get_tensor_value()
            ctx.remove_input(node, node.input[0], 0)
            node.set_attr("shape", shape)
            ctx.set_shape(node.output[0], shape)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        if node.inputs[0].is_const():
            cls.version_1(ctx, node, **kwargs)
        else:
            seed = node.get_attr("seed")
            node.set_attr("seed", float(seed.f))
            cast_node = ctx.make_node("Cast", node.input, attr={'to': onnx_pb.TensorProto.INT64})
            const_node = ctx.make_node("ConstantOfShape", cast_node.output)
            ctx.replace_inputs(node, const_node.output.copy())
            node.type = node.type + 'Like'


@tf_op(["RandomNormalLike", "RandomUniformLike"])
class PassThroughOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

@tf_op("Fill")
class Fill:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T output = Fill(int32 dims, T value, @int32 index_type)
        # T outputs = Tile(T value, int64 repeats (e.g. dims))
        fill_shape = ctx.get_shape(node.input[0])
        utils.make_sure(fill_shape is not None, "shape of {} is None".format(node.input[0]))
        fill_shape_dims = fill_shape[0]
        utils.make_sure(fill_shape_dims > 0, "opset 7 requires fill shape length > 0, or please try opset > 7")
        val_dtype = ctx.get_dtype(node.input[1])
        val_shape = ctx.get_shape(node.input[1])

        need_cast = val_dtype != onnx_pb.TensorProto.FLOAT and ctx.opset < 9
        new_dtype = val_dtype
        if need_cast:
            new_dtype = onnx_pb.TensorProto.FLOAT
            attr = {"to": new_dtype}
            cast_to_float = ctx.insert_new_node_on_input(node, "Cast", node.input[1], name=None, **attr)
            ctx.set_dtype(cast_to_float.output[0], new_dtype)
            ctx.set_shape(cast_to_float.output[0], val_shape)

        for _ in range(fill_shape_dims):
            attr = {"axes": [0]}
            shape = ctx.get_shape(node.input[1])
            unsqueeze_node = ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[1], name=None, **attr)
            ctx.set_dtype(unsqueeze_node.output[0], new_dtype)
            if shape:
                shape = [1] + shape
            else:
                shape = [1]
            ctx.set_shape(unsqueeze_node.output[0], shape)

        # Tile's repeats must be INT64
        attr = {"to": onnx_pb.TensorProto.INT64}
        tile_shape_int64 = ctx.insert_new_node_on_input(node, "Cast", node.input[0], name=None, **attr)
        ctx.set_dtype(tile_shape_int64.output[0], onnx_pb.TensorProto.INT64)
        ctx.set_shape(tile_shape_int64.output[0], fill_shape)

        tmp = node.input[0]
        ctx.replace_input(node, node.input[0], node.input[1], 0)
        ctx.replace_input(node, node.input[1], tmp, 1)
        node.type = "Tile"
        ctx.set_dtype(node.output[0], new_dtype)

        if need_cast:
            attr = {"to": val_dtype}
            op_name = utils.make_name(node.name + "/cast_back")
            cast_back = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name, **attr)
            ctx.set_dtype(cast_back.output[0], val_dtype)

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        node.type = "ConstantOfShape"
        # both shape and value in tensorflow are passed as tensor.
        # In onnx the value is an attribute so we need to fetch the value as const which
        # sooner or later will be a problem for tensorflow-onnx.
        # ConstantOfShape in onnxruntime only support int64, so insert cast op
        input_dtype_is_int64 = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[0])) == np.int64
        if not input_dtype_is_int64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=onnx_pb.TensorProto.INT64)
        dtype = ctx.get_dtype(node.output[0])
        value = np.array([node.inputs[1].get_tensor_value()]).astype(utils.map_onnx_to_numpy_type(dtype))
        value_proto = numpy_helper.from_array(value)
        node.set_attr("value", value_proto)
        ctx.remove_input(node, node.input[1], 1)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # cls.version_7(ctx, node, **kwargs)
        node.type = "Expand"
        ctx.replace_inputs(node, [node.input[1], node.input[0]])
        # cast shape to int64 if needed
        if ctx.get_dtype(node.input[1]) != onnx_pb.TensorProto.INT64:
            ctx.insert_new_node_on_input(node, "Cast", node.input[1], to=onnx_pb.TensorProto.INT64)


@tf_op("Multinomial")
class Multinomial:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # output_dtype output = Multinomial(T logits, int32 num_samples, @int seed, @int seed2, @type output_dtype)
        sample_size = node.inputs[1].get_tensor_value()
        seed = node.get_attr("seed")
        if seed:
            node.set_attr("seed", float(seed.i))
        output_dtype = node.get_attr("output_dtype")
        if output_dtype:
            output_dtype = output_dtype.i
        else:
            output_dtype = onnx_pb.TensorProto.INT32
        node.set_attr("dtype", output_dtype)
        node.set_attr("sample_size", sample_size)
        ctx.remove_input(node, node.input[1], 1)


@tf_op("ZerosLike")
class ZerosLike:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        casted_input = ctx.make_node("Cast", node.input, attr={'to': onnx_pb.TensorProto.INT64})
        const_zero = ctx.make_const(utils.make_name("zero"), np.array(0).astype(np.int64))
        mul_node = ctx.make_node('Mul', inputs=[casted_input.output[0], const_zero.output[0]])
        ctx.make_node("Cast", inputs=[mul_node.output[0]],
                      attr={'to': dtypes[0]},
                      name=node.name, outputs=node.output,
                      shapes=shapes, dtypes=dtypes)


@tf_op(["IteratorV2", "FIFOQueueV2"])
class Iterator:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        ctx.remove_node(node.name)


@tf_op(["IteratorGetNext", "QueueDequeueV2"])
class IteratorGetNext:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        output_names = node.output.copy()  # to make sure remove_node
                                           # does not alter the list
        type_0 = ctx.get_dtype(output_names[0])
        type_1 = ctx.get_dtype(output_names[1])
        shape_0 = ctx.get_shape(output_names[0])
        shape_1 = ctx.get_shape(output_names[1])
        ctx.remove_node(node.name)
        ctx.add_graph_input(output_names[0], type_0, shape_0)
        ctx.add_graph_input(output_names[1], type_1, shape_1)


@tf_op(["QueueDequeueManyV2", "QueueDequeueUpToV2"])
class QueueDequeueManyV2:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        outputs = node.output.copy()  # copy to make remove_node
                                      # does not alter the list
        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        for i, output in enumerate(outputs):
            ctx.add_graph_input(output, dtypes[i], shapes[i])
