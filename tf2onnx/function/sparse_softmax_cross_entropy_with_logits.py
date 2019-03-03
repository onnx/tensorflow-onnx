# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - sparse_softmax_cross_entropy_with_logits op conversion
"""
import numpy as np
from onnx.onnx_pb import TensorProto
from tf2onnx import utils
from tf2onnx.function.range import make_range
from tf2onnx.function.gathernd import make_gathernd

# pylint: disable=unused-argument,missing-docstring

def sparse_softmax_cross_entropy_with_logits_op(ctx, node, name, args):
    # make subgraph to implement one_hot, idea comes from onehot_op
    indices_name = node.input[1]
    indices_shape = ctx.get_shape(indices_name)
    if len(indices_shape) != 1:
        # TODO: this works for rank=1 but tensorflow supports more than this.
        # Same principle should work but we need to implement our own eye.
        raise ValueError("onehot op: only rank1 is supported")
    logit_name = node.input[0]
    depth = ctx.get_shape(logit_name)[-1]
    # if number of classes is unknown or too large
    if depth == utils.ONNX_UNKNOWN_DIMENSION or depth > 20000:
        sparse_softmax_cross_entropy_with_logits_op_by_gathernd(ctx, node, name, args)
        return
    logit_dtype = ctx.get_dtype(logit_name)
    utils.make_sure(logit_dtype, "Dtype of {} is None".format(logit_name))

    dtype = utils.map_onnx_to_numpy_type(logit_dtype)
    eye = np.eye(depth).astype(dtype)
    const_name = utils.make_name("const_eye")
    const_eye = ctx.make_const(name=const_name, np_val=eye)
    onehot = ctx.make_node(op_type="Gather", inputs=[const_eye.output[0], indices_name], attr={"axis": 0})
    log_softmax = ctx.make_node(op_type="LogSoftmax", inputs=[logit_name])
    # implement tf.multiply(np.float32(-1.0), tf.reduce_sum(tf.multiply(one_hot, log_softmax), axis=1))
    mul1 = ctx.make_node(op_type="Mul", inputs=[onehot.output[0], log_softmax.output[0]])
    reduce_sum = ctx.make_node(op_type="ReduceSum", inputs=[mul1.output[0]], attr={"axes": [1]})
    const_name = utils.make_name("const_negative_one")
    const_negative_one = ctx.make_const(name=const_name, np_val=np.array(-1).astype(dtype))
    mul2 = ctx.make_node(op_type="Mul", inputs=[const_negative_one.output[0], reduce_sum.output[0]])

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    ctx.make_node(op_type="Squeeze", inputs=[mul2.output[0]], outputs=[node.output[0]], attr={"axes": [1]},
                  shapes=[shapes[0]], dtypes=[dtypes[0]])


def sparse_softmax_cross_entropy_with_logits_op_by_gathernd(ctx, node, name, args):
    # make subgraph to implement one_hot, idea comes from onehot_op
    indices_name = node.input[1]
    indices_shape = ctx.get_shape(indices_name)
    if len(indices_shape) != 1:
        # TODO: this works for rank=1 but tensorflow supports more than this.
        # Same principle should work but we need to implement our own eye.
        raise ValueError("onehot op: only rank1 is supported")
    logit_name = node.input[0]
    logit_dtype = ctx.get_dtype(logit_name)
    utils.make_sure(logit_dtype, "Dtype of {} is None".format(logit_name))
    indices_dtype = ctx.get_dtype(indices_name)
    if indices_dtype != TensorProto.INT64:
        indices_cast = ctx.make_node("Cast", [indices_name], attr={"to": TensorProto.INT64})
        indices_name = indices_cast.output[0]
    indices_size = ctx.make_node("Size", [indices_name])
    indices_unsqueeze = ctx.make_node("Unsqueeze", [indices_name], attr={"axes": [1]})
    zero_const = ctx.make_const(utils.make_name("zero"), np.array(0, dtype=np.int64))
    one_const = ctx.make_const(utils.make_name("one"), np.array(1, dtype=np.int64))
    id_name = utils.make_name("sparse_softmax_id")
    id_output = utils.port_name(id_name)
    make_range(ctx, zero_const.output[0], indices_size.output[0], one_const.output[0],
               id_output, id_name, shape=[-1], dtype=TensorProto.INT64)
    id_unsqueeze = ctx.make_node("Unsqueeze", [id_output], attr={"axes": [1]})
    indices_with_id = ctx.make_node("Concat",
                                    [id_unsqueeze.output[0], indices_unsqueeze.output[0]],
                                    attr={"axis": 1})
    log_softmax = ctx.make_node(op_type="LogSoftmax", inputs=[logit_name], dtypes=[logit_dtype])
    gathernd_name = utils.make_name("sparse_softmax_gathernd")
    gathernd_output = utils.port_name(gathernd_name)
    make_gathernd(ctx, log_softmax.output[0], indices_with_id.output[0], gathernd_output,
                  gathernd_name, logit_dtype)
    const_name = utils.make_name("const_negative_one")
    const_negative_one = ctx.make_const(const_name, np.array(-1).astype(utils.map_onnx_to_numpy_type(logit_dtype)))
    mul2 = ctx.make_node(op_type="Mul", inputs=[const_negative_one.output[0], gathernd_output])
    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    ctx.make_node(op_type="Squeeze",
                  inputs=[mul2.output[0]], outputs=[node.output[0]],
                  attr={"axes": [1]}, shapes=[shapes[0]], dtypes=[dtypes[0]])
