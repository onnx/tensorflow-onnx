# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.utils - misc utilities for tf2onnx
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from onnx import helper, onnx_pb
from tensorflow.core.framework import types_pb2, tensor_pb2

#
#  mapping dtypes from tensorflow to onnx
#
TF_TO_ONNX_DTYPE = {
    types_pb2.DT_FLOAT: onnx_pb.TensorProto.FLOAT,
    types_pb2.DT_HALF: onnx_pb.TensorProto.FLOAT16,
    types_pb2.DT_DOUBLE: onnx_pb.TensorProto.DOUBLE,
    types_pb2.DT_INT32: onnx_pb.TensorProto.INT32,
    types_pb2.DT_INT16: onnx_pb.TensorProto.INT16,
    types_pb2.DT_INT8: onnx_pb.TensorProto.INT8,
    types_pb2.DT_UINT8: onnx_pb.TensorProto.UINT8,
    types_pb2.DT_UINT16: onnx_pb.TensorProto.UINT16,
    types_pb2.DT_INT64: onnx_pb.TensorProto.INT64,
    types_pb2.DT_STRING: onnx_pb.TensorProto.STRING,
    types_pb2.DT_COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    types_pb2.DT_COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
    types_pb2.DT_BOOL: onnx_pb.TensorProto.BOOL,
}

#
# mapping dtypes from onnx to numpy
#
ONNX_TO_NUMPY_DTYPE = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.FLOAT16: np.float16,
    onnx_pb.TensorProto.DOUBLE: np.float64,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.INT16: np.int16,
    onnx_pb.TensorProto.INT8: np.int8,
    onnx_pb.TensorProto.UINT8: np.uint8,
    onnx_pb.TensorProto.UINT16: np.uint16,
    onnx_pb.TensorProto.INT64: np.int64,
    onnx_pb.TensorProto.BOOL: np.bool,
}

#
#  onnx dtype names
#
ONNX_DTYPE_NAMES = {
    onnx_pb.TensorProto.FLOAT: "float",
    onnx_pb.TensorProto.FLOAT16: "float16",
    onnx_pb.TensorProto.DOUBLE: "double",
    onnx_pb.TensorProto.INT32: "int32",
    onnx_pb.TensorProto.INT16: "int16",
    onnx_pb.TensorProto.INT8: "int8",
    onnx_pb.TensorProto.UINT8: "uint8",
    onnx_pb.TensorProto.UINT16: "uint16",
    onnx_pb.TensorProto.INT64: "int64",
    onnx_pb.TensorProto.STRING: "string",
    onnx_pb.TensorProto.BOOL: "bool"
}

ONNX_UNKNOWN_DIMENSION = -1

#
# attributes onnx understands. Everything else coming from tensorflow
# will be ignored.
#
ONNX_VALID_ATTRIBUTES = {
    'p', 'bias', 'axes', 'pads', 'mean', 'activation_beta', 'spatial_scale', 'broadcast', 'pooled_shape', 'high',
    'activation_alpha', 'is_test', 'hidden_size', 'activations', 'beta', 'input_as_shape', 'drop_states', 'alpha',
    'momentum', 'scale', 'axis', 'dilations', 'transB', 'axis_w', 'blocksize', 'output_sequence', 'mode', 'perm',
    'min', 'seed', 'ends', 'paddings', 'to', 'gamma', 'width_scale', 'normalize_variance', 'group', 'ratio', 'values',
    'dtype', 'output_shape', 'spatial', 'split', 'input_forget', 'keepdims', 'transA', 'auto_pad', 'border', 'low',
    'linear_before_reset', 'height_scale', 'output_padding', 'shape', 'kernel_shape', 'epsilon', 'size', 'starts',
    'direction', 'max', 'clip', 'across_channels', 'value', 'strides', 'extra_shape', 'scales', 'k', 'sample_size',
    'blocksize', 'epsilon', 'momentum'
}


# index for internally generated names
INTERNAL_NAME = 1


def make_name(name):
    """Make op name for inserted ops."""
    global INTERNAL_NAME
    INTERNAL_NAME += 1
    return "{}__{}".format(name, INTERNAL_NAME)


def tf_to_onnx_tensor(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    new_type = TF_TO_ONNX_DTYPE[tensor.dtype]
    tdim = tensor.tensor_shape.dim
    dims = [d.size for d in tdim]
    # FIXME: something is fishy here
    if dims == [0]:
        dims = [1]
    is_raw, data = get_tf_tensor_data(tensor)
    onnx_tensor = helper.make_tensor(name, new_type, dims, data, is_raw)
    return onnx_tensor


def get_tf_tensor_data(tensor):
    """Get data from tensor."""
    assert isinstance(tensor, tensor_pb2.TensorProto)
    is_raw = False
    if tensor.tensor_content:
        data = tensor.tensor_content
        is_raw = True
    elif tensor.float_val:
        data = tensor.float_val
    elif tensor.dcomplex_val:
        data = tensor.dcomplex_val
    elif tensor.int_val:
        data = tensor.int_val
    elif tensor.bool_val:
        data = tensor.bool_val
    elif tensor.dtype == tf.int32:
        data = [0]
    elif tensor.dtype == tf.int64:
        data = [0]
    elif tensor.dtype == tf.float32:
        data = [0.]
    elif tensor.string_val:
        data = tensor.string_val
    else:
        raise ValueError('tensor data not supported')
    return [is_raw, data]


def get_shape(node):
    """Get shape from tensorflow node."""
    # FIXME: do we use this?
    dims = None
    try:
        if node.type == "Const":
            shape = node.get_attr("value").tensor_shape
            dims = [int(d.size) for d in shape.dim]
        else:
            shape = node.get_attr("shape")
            dims = [d.size for d in shape.dim]
        if shape[0] is not None or shape[0] == -1:
            shape[0] = 1
    except Exception as ex:
        pass
    return dims


def map_tf_dtype(dtype):
    if dtype:
        dtype = TF_TO_ONNX_DTYPE[dtype]
    return dtype


def node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name
