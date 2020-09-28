# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf_utils - misc utilities for tf2onnx that interface with tensorflow
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from tensorflow.core.framework import types_pb2, tensor_pb2
from tensorflow.python.framework import tensor_util

from onnx import helper, onnx_pb, numpy_helper

from tf2onnx.utils import make_sure, is_tf_const_op, port_name
from . import logging

logger = logging.getLogger(__name__)

#
#  mapping dtypes from tensorflow to onnx
#
TF_TO_ONNX_DTYPE = {
    types_pb2.DT_FLOAT: onnx_pb.TensorProto.FLOAT,
    types_pb2.DT_HALF: onnx_pb.TensorProto.FLOAT16,
    types_pb2.DT_BFLOAT16: onnx_pb.TensorProto.FLOAT16,
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
    types_pb2.DT_VARIANT: onnx_pb.TensorProto.UNDEFINED,
    types_pb2.DT_QUINT8: onnx_pb.TensorProto.UINT8,
}


def tf_to_onnx_tensor(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    np_data = get_tf_tensor_data(tensor)
    if np_data.dtype == np.object:
        # assume np_data is string, numpy_helper.from_array accepts ndarray,
        # in which each item is of str while the whole dtype is of object.
        try:
            if len(np_data.shape) > 0:
                np_data = np_data.astype(np.str).astype(np.object)
            else:
                np_data = np.array(str(np_data)).astype(np.object)
        except:  # pylint: disable=bare-except
            raise RuntimeError("Not support type: {}".format(type(np_data.flat[0])))
    return numpy_helper.from_array(np_data, name=name)


def get_tf_tensor_data(tensor):
    """Get data from tensor."""
    make_sure(isinstance(tensor, tensor_pb2.TensorProto), "Require TensorProto")
    np_data = tensor_util.MakeNdarray(tensor)
    make_sure(isinstance(np_data, np.ndarray), "%r isn't ndarray", np_data)
    return np_data


def get_tf_const_value(op, as_list=True):
    """
    If as_list=True, return the array as a (possibly nested) list.
    Otherwise, return data of type np.ndarray.

    If a tensor is a scalar having value 1,
        when as_list=False, return np.array(1), type is <class 'numpy.ndarray'>
        when as_list=True, return 1, type is <class 'int'>.
    """
    make_sure(is_tf_const_op(op), "%r isn't a const op", op.name)
    value = get_tf_tensor_data(op.get_attr("value"))
    if as_list:
        value = value.tolist()
    return value


def get_tf_shape_attr(node):
    """Get shape from tensorflow attr "shape"."""
    dims = None
    try:
        shape = get_tf_node_attr(node, "shape")
        if not shape.unknown_rank:
            dims = [int(d.size) for d in shape.dim]
    except:  # pylint: disable=bare-except
        pass
    return dims


def get_tf_tensor_shape(tensor):
    shape = []
    try:
        shape = tensor.get_shape().as_list()
    except Exception:  # pylint: disable=broad-except
        shape = None
    return shape


def map_tf_dtype(dtype):
    if dtype:
        dtype = TF_TO_ONNX_DTYPE[dtype]
    return dtype


def get_tf_node_attr(node, name):
    """Parser TF node attribute."""
    return node.get_attr(name)


def get_tf_version():
    return LooseVersion(tf.__version__)

def compress_graph_def(graph_def):
    """
    Remove large const values from graph. This lets us import the graph and run shape inference without TF crashing.
    """
    node_defs = list(graph_def.node)
    const_node_values = {}
    for node_def in node_defs:
        if node_def.op == 'Const':
            tensor = node_def.attr["value"].tensor
            # Small constants are sometimes used to store shape information and must be maintained
            if len(tensor.tensor_content) > 1000:
                make_sure(node_def.name not in const_node_values, "Two nodes in graph have same name %s", node_def.name)
                const_node_values[node_def.name] = tensor.tensor_content
                tensor.tensor_content = b''
    return const_node_values

def compute_const_folding_using_tf(g, const_node_values):
    """Find nodes with constant inputs and compute their values using TF"""
    if const_node_values is None:
        const_node_values = {}
    from tf2onnx.tf_loader import tf_session, tf_placeholder  # pylint: disable=import-outside-toplevel

    ops = g.get_operations()
    outputs_to_values = {}
    outputs_to_dtypes = {}

    for node in ops:
        # Load values of constants. Use const_node_values if possible
        if node.type in ["Const", "ConstV2"]:
            tensor = node.node_def.attr["value"].tensor
            if node.name in const_node_values:
                tensor.tensor_content = const_node_values[node.name]
            outputs_to_values[node.outputs[0].name] = get_tf_tensor_data(tensor)
            outputs_to_dtypes[node.outputs[0].name] = node.outputs[0].dtype

    unneeded_outputs = set()
    progress = True
    while progress:
        progress = False
        for node in ops:
            # Find ops with constant inputs and compute their values
            input_names = [i.name for i in node.inputs]
            output_names = [i.name for i in node.outputs]
            can_fold = node.type not in ['Enter']
            can_fold = can_fold and len(input_names) > 0 and all(inp in outputs_to_values for inp in input_names)
            # We can only fold nodes with a single output
            can_fold = can_fold and len(output_names) == 1 and output_names[0] not in outputs_to_values
            # Skip if value already computed, used, and discarded
            can_fold = can_fold and output_names[0] not in unneeded_outputs
            if can_fold:
                # Make a mini graph containing just the node to fold
                g2 = tf.Graph()
                with g2.as_default():
                    for inp in input_names:
                        tf_placeholder(outputs_to_dtypes[inp], name=inp.split(':')[0])
                    mini_graph_def = g2.as_graph_def()
                    mini_graph_def.node.append(node.node_def)
                g3 = tf.Graph()
                with g3.as_default():
                    feed_dict = {}
                    for inp in input_names:
                        feed_dict[inp] = outputs_to_values[inp]
                    try:
                        with tf_session() as sess:
                            tf.import_graph_def(mini_graph_def, name='')
                            results = sess.run(output_names, feed_dict=feed_dict)
                        outputs_to_values[output_names[0]] = results[0]
                        outputs_to_dtypes[output_names[0]] = node.outputs[0].dtype
                        progress = True
                    except Exception:  # pylint: disable=broad-except
                        logger.debug("Could not fold node %s", node.name)
        unneeded_outputs.update(outputs_to_values.keys())
        for node in ops:
            # Mark values we need to keep
            input_names = [i.name for i in node.inputs]
            output_names = [i.name for i in node.outputs]
            if len(output_names) == 1 and output_names[0] in outputs_to_values:
                continue
            for i in input_names:
                if i in unneeded_outputs:
                    unneeded_outputs.remove(i)
        for node in unneeded_outputs:
            # Remove unneeded values to prevent memory usage explosion
            if node in outputs_to_values:
                del outputs_to_values[node]
                del outputs_to_dtypes[node]

    for node in ops:
        # We don't need the constants any more
        if node.type in ["Const", "ConstV2"] and node.outputs[0].name in outputs_to_values:
            del outputs_to_values[node.outputs[0].name]
            del outputs_to_dtypes[node.outputs[0].name]

    logger.info("Computed %d values for constant folding", len(outputs_to_values))
    return outputs_to_values, outputs_to_dtypes

def tflist_to_onnx(g, shape_override, const_node_values=None):
    """
    Convert the tf-node list into an onnx graph with minimal rewrites so
    we can use the onnx graph as intermediate graph.
    """

    # ignore the following attributes
    ignored_attr = {"unknown_rank", "_class", "Tshape", "use_cudnn_on_gpu", "Index", "Tpaddings",
                    "TI", "Tparams", "Tindices", "Tlen", "Tdim", "Tin", "dynamic_size", "Tmultiples",
                    "Tblock_shape", "Tcrops", "index_type", "Taxis", "U", "maxval",
                    "Tout", "Tlabels", "Tindex", "element_shape", "Targmax", "Tperm", "Tcond",
                    "T_threshold", "element_dtype", "shape_type", "_lower_using_switch_merge",
                    "parallel_iterations", "_num_original_outputs", "output_types", "output_shapes",
                    "key_dtype", "value_dtype", "Tin", "Tout", "capacity", "component_types", "shapes",
                    "Toutput_types",
                    "Tcomplex", "Treal",  # For RFFT, Tcomplex is ignored because
                                          # onnx.helper.make_node fails,
                                          # TODO: it should be added back.
                    }

    node_list = g.get_operations()
    functions = {}

    # some stats
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}

    # find outputs
    ops = node_list

    # create dict with output to shape mappings
    for node in ops:
        for out in node.outputs:
            shape = shape_override.get(out.name)
            if shape is None:
                shape = get_tf_tensor_shape(out)
            dtypes[out.name] = map_tf_dtype(out.dtype)
            output_shapes[out.name] = shape

    # minimal conversion of attributes
    for node in ops:
        attr = {}
        takeit = True
        op_cnt[node.type] += 1
        for a in node.node_def.attr:
            attr_cnt[a] += 1
            if a == "dtype":
                attr[a] = map_tf_dtype(get_tf_node_attr(node, "dtype"))
            elif a == "T":
                dtype = get_tf_node_attr(node, a)
                if dtype and not isinstance(dtype, list):
                    dtypes[node.name] = map_tf_dtype(dtype)
            elif a in {"output_type", "output_dtype", "out_type", "Tidx", "out_idx"}:
                # Tidx is used by Range
                # out_idx is used by ListDiff
                attr[a] = map_tf_dtype(get_tf_node_attr(node, a))
            elif a == "shape":
                shape = get_tf_shape_attr(node)
                if shape is not None:
                    attr[a] = shape
            elif a == "output_shapes":
                # we should not need it since we pull the shapes above already
                pass
            elif a in {"body", "cond", "then_branch", "else_branch", "f"}:
                input_shapes = [inp.get_shape() for inp in node.inputs]
                nattr = get_tf_node_attr(node, a)
                attr[a] = nattr.name
                functions[nattr.name] = input_shapes
            elif a == "value":
                tensor = get_tf_node_attr(node, a)
                if const_node_values and node.name in const_node_values:
                    tensor.tensor_content = const_node_values[node.name]
                onnx_tensor = tf_to_onnx_tensor(tensor, name=port_name(node.name))
                attr[a] = onnx_tensor
            elif a == "DstT":
                attr["to"] = map_tf_dtype(get_tf_node_attr(node, "DstT"))
            elif a == "SrcT":
                continue
            elif a in ignored_attr:
                continue
            else:
                attr[a] = get_tf_node_attr(node, a)

        if takeit:
            try:
                input_names = [i.name for i in node.inputs]
                output_names = [i.name for i in node.outputs]
                onnx_node = helper.make_node(node.type, input_names, output_names, name=node.name, **attr)
                onnx_nodes.append(onnx_node)
            except Exception as ex:
                logger.error("pass1 convert failed for %s, ex=%s", node, ex)
                raise

    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, functions


def tensorflow_to_onnx(graph, shape_override, const_node_values=None):
    """
    Load tensorflow graph and do a conversion.
    """
    return tflist_to_onnx(graph, shape_override, const_node_values)
