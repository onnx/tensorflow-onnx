# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.utils - misc utilities for tf2onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import shutil
import tempfile
import six
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import types_pb2, tensor_pb2
from google.protobuf import text_format
from onnx import helper, onnx_pb, defs, numpy_helper

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
    types_pb2.DT_RESOURCE: onnx_pb.TensorProto.INT64,  # TODO: hack to allow processing on control flow
    types_pb2.DT_QUINT8: onnx_pb.TensorProto.UINT8,  # TODO: map quint8 to  uint8 for now
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
    'blocksize', 'epsilon', 'momentum', 'body', 'directions', 'num_scan_inputs', 'then_branch', 'else_branch'
}

# index for internally generated names
INTERNAL_NAME = 1

# Fake onnx op type which is used for Graph input.
GRAPH_INPUT_TYPE = "NON_EXISTENT_ONNX_TYPE"


def make_name(name):
    """Make op name for inserted ops."""
    global INTERNAL_NAME
    INTERNAL_NAME += 1
    return "{}__{}".format(name, INTERNAL_NAME)


def split_nodename_and_shape(name):
    """input name with shape into name and shape."""
    # pattern for a node name
    inputs = []
    shapes = {}
    # input takes in most cases the format name:0, where 0 is the output number
    # in some cases placeholders don't have a rank which onnx can't handle so we let uses override the shape
    # by appending the same, ie : [1,28,28,3]
    name_pattern = r"(?:([\w\d/\-\._:]+)(\[[\-\d,]+\])?),?"
    splits = re.split(name_pattern, name)
    for i in range(1, len(splits), 3):
        inputs.append(splits[i])
        if splits[i + 1] is not None:
            shapes[splits[i]] = [int(n) for n in splits[i + 1][1:-1].split(",")]
    if not shapes:
        shapes = None
    return inputs, shapes


def tf_to_onnx_tensor(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    new_type = TF_TO_ONNX_DTYPE[tensor.dtype]
    tdim = tensor.tensor_shape.dim
    dims = [d.size for d in tdim]
    # FIXME: something is fishy here
    if dims == [0]:
        dims = [1]
    is_raw, data = get_tf_tensor_data(tensor)
    if not is_raw and len(data) == 1 and np.prod(dims) > 1:
        batch_data = np.zeros(dims, dtype=ONNX_TO_NUMPY_DTYPE[new_type])
        batch_data.fill(data[0])
        onnx_tensor = numpy_helper.from_array(batch_data, name=name)
    else:
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
    elif tensor.int64_val:
        data = tensor.int64_val
    elif tensor.bool_val:
        data = tensor.bool_val
    elif tensor.dtype == tf.int32:
        data = [0]
    elif tensor.dtype == tf.int64:
        data = [0]
    elif tensor.dtype == tf.float32:
        data = [0.]
    elif tensor.dtype == tf.float16:
        data = [0]
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
            shape = get_tf_node_attr(node, "value").tensor_shape
            dims = [int(d.size) for d in shape.dim]
        else:
            shape = get_tf_node_attr(node, "shape")
            dims = [d.size for d in shape.dim]
    except:  # pylint: disable=bare-except
        pass
    return dims


def map_tf_dtype(dtype):
    if dtype:
        dtype = TF_TO_ONNX_DTYPE[dtype]
    return dtype


def map_numpy_to_onnx_dtype(np_dtype):
    for onnx_dtype, numpy_dtype in ONNX_TO_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported dtype " + np_dtype + " for mapping")


def node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def make_onnx_shape(shape):
    """shape with -1 is not valid in onnx ... make it a name."""
    if shape:
        # don't do this if input is a scalar
        return [make_name("unk") if i == -1 else i for i in shape]
    return shape


def port_name(name, nr=0):
    """Map node output number to name."""
    return name + ":" + str(nr)


def make_onnx_identity(node_input, node_output, name=None):
    if name is None:
        name = make_name("identity")
    return helper.make_node("Identity", [node_input], [node_output], name=name)


def make_onnx_inputs_outputs(name, elem_type, shape, **kwargs):
    """Wrapper for creating onnx graph inputs or outputs
       name,  # type: Text
       elem_type,  # type: TensorProto.DataType
       shape,  # type: Optional[Sequence[int]]
    """
    return helper.make_tensor_value_info(name, elem_type, make_onnx_shape(shape), **kwargs)


PREFERRED_OPSET = 7


def find_opset(opset):
    """Find opset."""
    if opset is None or opset == 0:
        opset = defs.onnx_opset_version()
        if opset > PREFERRED_OPSET:
            # if we use a newer onnx opset than most runtimes support, default to the one most supported
            opset = PREFERRED_OPSET
    return opset


def get_tf_node_attr(node, name):
    """Parser TF node attribute."""
    if six.PY2:
        # For python2, TF get_attr does not accept unicode
        name = str(name)
    return node.get_attr(name)


def save_onnx_model(save_path_root, onnx_file_name, feed_dict, model_proto, include_test_data=False, as_text=False):
    """Save onnx model as file. Save a pbtxt file as well if as_text is True"""
    save_path = save_path_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if include_test_data:
        data_path = os.path.join(save_path, "test_data_set_0")
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        i = 0
        for data_key in feed_dict:
            data = feed_dict[data_key]
            t = numpy_helper.from_array(data)
            t.name = data_key
            data_full_path = os.path.join(data_path, "input_" + str(i) + ".pb")
            save_protobuf(data_full_path, t)
            i += 1

    target_path = os.path.join(save_path, onnx_file_name + ".onnx")
    save_protobuf(target_path, model_proto)
    if as_text:
        save_protobuf(target_path + ".pbtxt", model_proto, as_text=True)
    return target_path


def make_sure(bool_val, error_msg, *args):
    if not bool_val:
        raise ValueError("make_sure failure: " + error_msg % args)


def construct_graph_from_nodes(parent_g, nodes, outputs, shapes, dtypes):
    """Construct Graph from nodes and outputs with specified shapes and dtypes."""
    # pylint: disable=protected-access
    g = parent_g.create_new_graph_with_same_config()
    g.parent_graph = parent_g
    nodes = set(nodes)
    all_outputs = set()
    ops = []
    for op in nodes:
        all_outputs |= set(op.output)

        new_node = g.make_node(op.type, op.input, outputs=op.output, attr=op.attr, name=op.name,
                               skip_conversion=op._skip_conversion)
        body_graphs = op.graph.contained_graphs.pop(op.name, None)
        if body_graphs:
            for attr_name, body_graph in body_graphs.items():
                body_graph.parent_graph = g
                new_node.set_body_graph_as_attr(attr_name, body_graph)
        ops.append(new_node)

    for i in all_outputs:
        if i not in g._output_shapes:
            g._output_shapes[i] = parent_g._output_shapes[i]
        if i not in g._dtypes:
            g._dtypes[i] = parent_g._dtypes[i]

    g.set_nodes(ops)

    # handle cell graph: insert identity node, since sometimes we need output same output_id
    # as state_output and scan_out, but ONNX don't allow the same output_id to appear more
    # than once as output node.
    cell_body_nodes = []
    new_output_names = []
    for output, shape, dtype in zip(outputs, shapes, dtypes):
        node = g.make_node("Identity", inputs=[output], op_name_scope="sub_graph_ending_node",
                           shapes=[shape], dtypes=[dtype])
        new_output_names.append(node.output[0])
        cell_body_nodes.append(node)

    cell_nodes = g.get_nodes()
    cell_nodes.extend(cell_body_nodes)
    g.set_nodes(cell_nodes)
    g.outputs = new_output_names
    return g


def tf_name_scope(name):
    return '/'.join(name.split('/')[:-1])


def create_vague_shape_like(shape):
    make_sure(len(shape) >= 0, "rank should be >= 0")
    return [-1 for i in enumerate(shape)]


def get_temp_directory():
    return os.environ.get("TF2ONNX_TEMP_DIRECTORY", tempfile.mkdtemp())


def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_protobuf(path, message, as_text=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())
