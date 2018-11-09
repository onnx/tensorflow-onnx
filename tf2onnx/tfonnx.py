# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - rewrite tensorflow graph to onnx graph
"""

from __future__ import division
from __future__ import print_function

import collections
import logging
import sys
import traceback

import numpy as np
from onnx import helper, onnx_pb, numpy_helper

from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

import tf2onnx
from tf2onnx import utils
from tf2onnx.graph import Node, Graph
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.rewriter.rnn import rewrite_single_direction_lstm, rewrite_bi_direction_lstm
from tf2onnx.rewriter.rnn import rewrite_single_direction_gru
from tf2onnx.rewriter.rnn import rewrite_single_direction_grublock
from tf2onnx.rewriter.rnn import rewrite_bi_direction_gru

from tf2onnx.utils import port_name

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx")

# Target for the generated onnx graph. It possible targets:
# onnx-1.1 = onnx at v1.1 (winml in rs4 is based on this)
# caffe2 = include some workarounds for caffe2 and winml
TARGET_RS4 = "rs4"
TARGET_RS5 = "rs5"
TARGET_CAFFE2 = "caffe2"
POSSIBLE_TARGETS = [TARGET_RS4, TARGET_RS5, TARGET_CAFFE2]
DEFAULT_TARGET = []

# pylint: disable=useless-return,broad-except,logging-not-lazy,unused-argument,missing-docstring
# FIXME:
# pylint: disable=unused-variable


def tflist_to_onnx(node_list, shape_override):
    """
    Convert the tf-node list into an onnx graph with minimal rewrites so
    we can use the onnx graph as intermediate graph.
    """

    # ignore the following attributes
    ignored_attr = ["unknown_rank", "_class", "Tidx", "Tshape", "use_cudnn_on_gpu", "Index",
                    "Tpaddings", "TI", "Tparams", "Tindices", "Tlen", "Tdim", "dynamic_size", "element_shape",
                    "Tmultiples", "output_dtype", "Tblock_shape", "Tcrops", "index_type", "Taxis", "U",
                    "maxval", "Tout"]
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
                try:
                    shape = out.get_shape().as_list()
                except Exception as ex:
                    shape = []
            dtypes[out.name] = utils.map_tf_dtype(out.dtype)
            output_shapes[out.name] = shape

    # minimal conversion of attributes
    for node in ops:
        attr = {}
        takeit = True
        op_cnt[node.type] += 1
        for a in node.node_def.attr:
            attr_cnt[a] += 1
            if a == "dtype":
                attr[a] = utils.map_tf_dtype(node.get_attr("dtype"))
            elif a == "T":
                dtype = node.get_attr("T")
                if dtype:
                    if not isinstance(dtype, list):
                        dtypes[node.name] = utils.map_tf_dtype(dtype)
            elif a in ["output_type", "output_dtype", "out_type"]:
                attr[a] = utils.map_tf_dtype(node.get_attr(a))
            elif a == "shape":
                attr[a] = utils.get_shape(node)
            elif a == "Tperm":
                pass
            elif a == "_output_shapes":
                attr[a] = utils.get_shape(node)
            elif a == "value":
                onnx_tensor = utils.tf_to_onnx_tensor(node.get_attr(a), name=port_name(node.name))
                attr[a] = onnx_tensor
            elif a == "DstT":
                attr["to"] = utils.map_tf_dtype(node.get_attr("DstT"))
            elif a == "SrcT":
                continue
            elif a in ignored_attr:
                continue
            else:
                attr[a] = node.get_attr(a)

        if takeit:
            try:
                input_names = [i.name for i in node.inputs]
                output_names = [i.name for i in node.outputs]
                onnx_node = helper.make_node(node.type, input_names, output_names, name=node.name, **attr)
                onnx_nodes.append(onnx_node)
            except Exception as ex:
                log.error("pass1 convert failed for %s, ex=%s", node, ex)
                raise

    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes


def tensorflow_to_onnx(graph, shape_override):
    """
    Load tensorflow graph and do a conversion.
    """
    return tflist_to_onnx(graph.get_operations(), shape_override)


def _convert_shapenode_to_int64(ctx, node, input_number):
    """cast int32 shape into int64 shape."""
    shape_node = node.inputs[input_number]
    name = node.input[input_number]
    if shape_node.is_const():
        # if it is a const, change the const to be int64
        shape = shape_node.get_tensor_value()
        shape = np.array(list(shape), dtype=np.int64)
        onnx_tensor = numpy_helper.from_array(shape, name)
        ctx.set_initializer(name, onnx_tensor)
        shape_node.set_attr("value", onnx_tensor)
        ctx.set_dtype(shape_node.output[0], onnx_pb.TensorProto.INT64)
        ctx.copy_shape(name, shape_node.output[0])
        return [node]

    cast_node = ctx.insert_new_node_on_input(node, "Cast", name)
    cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
    ctx.copy_shape(name, cast_node.output[0])
    return [cast_node, node]


def no_op(ctx, node, name, args):
    """Skip node."""
    return None


def direct_op(ctx, node, name, args):
    """Take node as is, no updates required"""
    return node


def identity_op(ctx, node, name, args):
    """Identity."""
    if node.inputs[0].is_const():
        # if identity has a const as input, remove it
        input_name = node.input[0]
        output_name = node.output[0]
        for n in ctx.get_nodes():
            for i, parent_name in enumerate(n.input):
                if parent_name == output_name:
                    n.input[i] = input_name
        return None
    ctx.copy_shape(node.input[0], node.output[0])
    return node


def broadcast_op(ctx, node, name, args):
    """Elementwise Ops with broadcast flag."""
    shape0 = ctx.get_shape(node.input[0])
    shape1 = ctx.get_shape(node.input[1])
    if shape0 != shape1:
        node.set_attr("broadcast", 1)
        # this works around shortcomings in the broadcasting code
        # of caffe2 and winml/rs4.
        if ctx.is_target(TARGET_RS4):
            # in rs4 mul and add do not support scalar correctly
            if not shape0:
                if node.inputs[0].is_const():
                    shape0 = node.inputs[0].scalar_to_dim1()
            if not shape1:
                if node.inputs[1].is_const():
                    shape1 = node.inputs[1].scalar_to_dim1()
        if shape0 and shape1 and len(shape0) < len(shape1) and node.type in ["Mul", "Add"]:
            tmp = node.input[0]
            node.input[0] = node.input[1]
            node.input[1] = tmp
    else:
        node.set_attr("broadcast", 0)
    return node


def broadcast_op7(ctx, node, name, args):
    """Elementwise Ops with broadcast flag."""
    shape0 = ctx.get_shape(node.input[0])
    shape1 = ctx.get_shape(node.input[1])
    if shape0 != shape1:
        # this works around shortcomings in the broadcasting code
        # of caffe2 and winml/rs4.
        if ctx.is_target(TARGET_RS4):
            # in rs4 mul and add do not support scalar correctly
            if not shape0:
                if node.inputs[0].is_const():
                    shape0 = node.inputs[0].scalar_to_dim1()
            if not shape1:
                if node.inputs[1].is_const():
                    shape1 = node.inputs[1].scalar_to_dim1()
        if shape0 and shape1 and len(shape0) < len(shape1) and node.type in ["Mul", "Add"]:
            tmp = node.input[0]
            node.input[0] = node.input[1]
            node.input[1] = tmp
    return node


def const_op(ctx, node, name, args):
    """Constants - make those initializers."""
    tensor = node.get_attr("value")
    ctx.add_initializer(tensor.t)
    # we return None - const will not be in the node list. But we keep the mapping for
    # get_node_by_name() so we don't need to lookup the initializers.
    return None


def arg_minmax_op(ctx, node, name, args):
    # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
    # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
    axis_node = node.inputs[1]
    axis = axis_node.get_tensor_value()
    node.set_attr("axis", axis[0])
    ctx.remove_input(node, node.input[1])
    return node


def reduce_op(ctx, node, name, args):
    axes_node = node.inputs[1]
    axis = axes_node.get_tensor_value()
    node.set_attr("axes", axis)
    ctx.remove_input(node, node.input[1])
    keep_dims = node.get_attr("keep_dims")
    if keep_dims:
        del node.attr['keep_dims']
        node.set_attr("keepdims", keep_dims.i)
    return node


def placeholder_op(ctx, node, name, args):
    input_node = helper.make_tensor_value_info(node.output[0],
                                               node.dtype,
                                               utils.make_onnx_shape(ctx.get_shape(node.output[0])))
    ctx.add_model_input(input_node.name, input_node)
    return None


def square_op(ctx, node, name, args):
    node.type = "Mul"
    node.input.append(node.input[0])
    return node


def squeeze_op(ctx, node, name, args):
    # T output = Squeeze(T input, @list(int) squeeze_dims)
    # T squeezed = Squeeze(T data, @AttrType.INTS axes)
    axis = node.get_attr("axis")
    if not axis:
        axis = node.get_attr("squeeze_dims")
        if axis:
            del node.attr["squeeze_dims"]
    else:
        del node.attr["axis"]

    shape = ctx.get_shape(node.input[0])
    if axis and axis.ints:
        axis = axis.ints
    else:
        axis = [i for i, j in enumerate(shape) if j == 1]
    node.set_attr("axes", axis)
    return node


def reshape_op(ctx, node, name, args):
    # T output = Reshape(T tensor, Tshape shape, @type Tshape)
    # T reshaped = Reshape(T data, @INTS shape) - but takes a optional 2nd input for shape
    shape_node = node.inputs[1]
    shape = shape_node.get_tensor_value()
    if shape is None:
        log.error("Reshape on node %s does not have a const shape", node.name)
        return None
    ctx.remove_input(node, node.input[1])
    node.set_attr("shape", shape)
    ctx.set_shape(node.output[0], shape)
    return node


def reshape_op5(ctx, node, name, args):
    need_casting = node.dtype in [onnx_pb.TensorProto.INT32,
                                  onnx_pb.TensorProto.INT16,
                                  onnx_pb.TensorProto.INT64]
    # onnx wants reshape.input[1] to have the value be int64 which is not the case for tensorflow.
    nodes = _convert_shapenode_to_int64(ctx, node, 1)
    if not need_casting:
        # onnx reshape can handle the type - done
        return nodes

    # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
    input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
    input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
    ctx.copy_shape(name, input_cast.output[0])

    # if the next node is already a cast we don't need to insert another one
    next_nodes = ctx.find_output_consumers(node.output[0])
    if len(next_nodes) != 1 or next_nodes[0].type != "Cast":
        op_name = utils.make_name(node.name)
        output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name)
        output_cast.set_attr("to", node.dtype)
        ctx.set_dtype(output_cast.output[0], node.dtype)
        ctx.copy_shape(name, output_cast.output[0])
        nodes.append(output_cast)
    return [input_cast] + nodes


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]


def spatial_map(shape, perm):
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def conv_convert_inputs(ctx, node, with_kernel=False, new_kernel_shape=None,
                        input_indices=None, output_indices=None):
    """Convert input and kernel from tensorflow to onnx. This maybe require to
        to insert transpose ops for input, kernel and output unless they are constants
        and we can transpose the constant.
        We transpose inputs if they are in NHWC. We always transpose the kernel from
        HWNC to NCHW. Outputs are transposed if the format is NHWC.
        Some convolutions like depthwise_conv2d require a reshape of the kernel.
        Args:
            ctx: the parent graph
            node: node of the convolution op
            with_kernel: transpose the kernel
            new_kernel_shape: reshape the kernel
    """

    if input_indices is None:
        input_indices = [0]
    if output_indices is None:
        output_indices = [0]

    nodes = []

    if node.is_nhwc():
        # transpose input if needed, no need to record shapes on input
        for idx in input_indices:
            parent = node.inputs[idx]
            if node.inputs[idx].is_const():
                # if input is a constant, transpose that one
                if not parent.data_format:
                    val = parent.get_tensor_value()
                    parent.set_tensor_value(val.transpose(NHWC_TO_NCHW))
            else:
                # if input comes from a op, insert transpose op
                input_name = node.input[idx]
                transpose = ctx.insert_new_node_on_input(node, "Transpose", input_name)
                transpose.set_attr("perm", NHWC_TO_NCHW)
                transpose.inserted_nchw = True
                shape = ctx.get_shape(input_name)
                new_shape = spatial_map(shape, NHWC_TO_NCHW)
                ctx.set_shape(transpose.output[0], new_shape)
                nodes.append(transpose)
            parent.data_format = "NCHW"

    # kernel must to be transposed
    if with_kernel:
        parent = node.inputs[1]
        if node.inputs[1].is_const():
            # kernel is const - transpose the const
            if not parent.data_format:
                val = parent.get_tensor_value()
                val = val.transpose(HWCN_TO_NCHW)
                parent.set_tensor_value(val)
        else:
            # kernel comes from op, insert transpose op
            input_name = node.input[1]
            transpose = ctx.insert_new_node_on_input(node, "Transpose", input_name)
            transpose.set_attr("perm", HWCN_TO_NCHW)
            transpose.inserted_nchw = True
            ctx.copy_shape(input_name, transpose.output[0])
            new_shape = spatial_map(ctx.get_shape(input_name), HWCN_TO_NCHW)
            ctx.set_shape(transpose.output[0], new_shape)
            nodes.append(transpose)
        parent.data_format = "NCHW"

        # some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
        if new_kernel_shape:
            if ctx.opset < 5:
                # old reshape takes new shape as attribute
                input_name = node.input[1]
                reshape = ctx.insert_new_node_on_input(node, "Reshape", input_name)
                reshape.set_attr("shape", new_kernel_shape)
            else:
                # new reshape takes new shape as input[1]
                shape_name = utils.make_name(node.name)
                ctx.make_const(shape_name, np.array(new_kernel_shape, dtype=np.int64))
                input_name = node.input[1]
                reshape = ctx.insert_new_node_on_input(node, "Reshape", input_name)
                reshape.input.append(shape_name)
            ctx.set_shape(reshape.output[0], new_kernel_shape)
            nodes.append(reshape)

    # insert conv node after inputs
    nodes.append(node)

    # transpose outputs if needed
    if node.is_nhwc():
        for idx in output_indices:
            output_name = node.output[idx]
            op_name = utils.make_name(node.name)
            transpose = ctx.insert_new_node_on_output("Transpose", output_name, name=op_name)
            transpose.set_attr("perm", NCHW_TO_NHWC)
            transpose.inserted_nchw = True
            ctx.set_shape(transpose.output[0], ctx.get_shape(node.output[idx]))
            nodes.append(transpose)
            node.data_format = "NCHW"
    return nodes


def add_padding(ctx, node, kernel_shape, strides, dilations=None, spatial=2):
    padding = node.get_attr("padding")
    if padding:
        if dilations is None:
            dilations = [1] * spatial * 2
        padding = padding.s.decode("utf-8")
        if padding == 'SAME':
            pads = [0] * spatial * 2
            input_shape = ctx.get_shape(node.input[0])
            output_shape = ctx.get_shape(node.output[0])
            # check if the input shape is valid
            if len(input_shape) != len(pads):
                log.error("node %s input needs to be rank %d, is %d" % (node.name, len(pads), len(input_shape)))
            # transpose shape to nchw
            if node.is_nhwc():
                input_shape = spatial_map(input_shape, NHWC_TO_NCHW)
                output_shape = spatial_map(output_shape, NHWC_TO_NCHW)
            # calculate pads
            for i in range(spatial):
                if input_shape[i + 2] == -1:
                    log.error("node %s has unknown dim %s for pads calculation" % (node.name, str(input_shape)))
                    continue
                pad = (output_shape[i + 2] - 1) * strides[i] + dilations[i] * kernel_shape[i] - input_shape[i + 2]
                pad = max(pad, 0)
                pads[i] = pad // 2
                pads[i + spatial] = pad - pad // 2
            node.set_attr("pads", pads)

        elif padding == 'VALID':
            pass
        else:
            raise ValueError("invalid padding value: " + padding)


def conv_dims_attr(node, name, new_name=None):
    if new_name is None:
        new_name = name
    dims = node.get_attr(name)
    if not dims:
        return None
    dims = dims.ints
    if node.is_nhwc():
        if len(dims) == 2:
            h, w = dims
            c = n = 1
        else:
            n, h, w, c = dims
    else:
        n, c, h, w = dims
    dims = [h, w]
    node.set_attr(new_name, dims)
    return dims


def conv_kernel_shape(ctx, node, input_idx, spatial=2):
    kernel_shape = ctx.get_shape(node.input[input_idx])
    if len(kernel_shape) != 2 * spatial:
        raise ValueError("kernel rank must be 2* spatial")
    kernel_shape = kernel_shape[0:spatial]
    node.set_attr("kernel_shape", kernel_shape)
    return kernel_shape


def conv_op(ctx, node, name, args):
    # T output = Conv2D(T input, T filter, @list(int) strides, @bool use_cudnn_on_gpu,
    #                       @string padding, @string data_format)
    # T Y = Conv(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
    #                       @AttrType.INTS kernel_shape, @AttrType.INTS pads, @AttrType.INTS strides)
    kernel_shape = conv_kernel_shape(ctx, node, 1, spatial=2)
    strides = conv_dims_attr(node, "strides")
    dilations = conv_dims_attr(node, "dilations")
    add_padding(ctx, node, kernel_shape, strides, dilations=dilations, spatial=2)
    nodes = conv_convert_inputs(ctx, node, with_kernel=True)
    return nodes


def convtranspose_op(ctx, node, name, args):
    # T output = Conv2DBackpropInput(int32 input_sizes, T filter, T out_backprop,
    #    @list(int) strides, @bool use_cudnn_on_gpu, @string padding, @string data_format, @list(int) dilations)
    # T Y = ConvTranspose(T X, T W, T B, @STRING auto_pad, @INTS dilations,
    #    @INT group, @INTS kernel_shape, @INTS output_shape, @INTS pads, @INTS strides)

    # Note: inputs are reversed from what one would expect.
    kernel_shape = conv_kernel_shape(ctx, node, 1)

    # ouput_shape is explicitly specified here, in this case pads values are auto generated/calculated.
    output_shape = ctx.get_shape(node.output[0])
    if node.is_nhwc():
        new_output_shape = [output_shape[1], output_shape[2]]
    else:
        new_output_shape = [output_shape[2], output_shape[3]]
    node.set_attr("output_shape", new_output_shape)

    strides = conv_dims_attr(node, "strides")
    conv_dims_attr(node, "dilations")

    # remove output_shapes input
    ctx.remove_input(node, node.input[0])
    # swap data and kernel
    t = node.input[0]
    node.input[0] = node.input[1]
    node.input[1] = t

    nodes = conv_convert_inputs(ctx, node, with_kernel=True)

    # Note: output_padding, group are left default.
    return nodes


def depthwiseconv_op(ctx, node, name, args):
    # T output = DepthwiseConv2dNative(T input, T filter, @list(int) strides, @string padding, @string data_format)
    # T Y = ConvTranspose(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
    #           @AttrType.INTS kernel_shape, @AttrType.INTS output_shape, @AttrType.INTS pads, @AttrType.INTS strides)
    #
    # this is not documented well in onnx, the hint comes from pytorch documentation:
    # http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    #   The configuration when groups == in_channels and out_channels = K * in_channels
    #   where K is a positive integer is termed in literature as depthwise convolution.
    #   In other words, for an input of size (N,Cin,Hin,Win),
    #   if you want a depthwise convolution with a depthwise multiplier K,
    #   then you use the constructor arguments (in_channels=Cin,out_channels=Cin*K,...,groups=Cin)
    #
    input_shape = ctx.get_shape(node.input[0])
    if len(input_shape) != 4:
        raise ValueError("only Conv2D is supported")

    if node.is_nhwc():
        i_n, i_h, i_w, i_c = input_shape
    else:
        i_n, i_c, i_h, i_w = input_shape

    kernel_shape = ctx.get_shape(node.input[1])
    if len(kernel_shape) != 4:
        raise ValueError("only Conv2D is supported")
    k_h, k_w, k_input_channels, k_channel_multiplier = kernel_shape
    k_output_channels = i_c * k_channel_multiplier

    node.set_attr("kernel_shape", [k_h, k_w])
    strides = conv_dims_attr(node, "strides")
    conv_dims_attr(node, "dilations")
    node.set_attr("group", i_c)
    add_padding(ctx, node, kernel_shape, strides)

    new_kernel_shape = [k_output_channels, 1, k_h, k_w]
    nodes = conv_convert_inputs(ctx, node, with_kernel=True, new_kernel_shape=new_kernel_shape)
    return nodes


def pool_op(ctx, node, name, args):
    # T output = MaxPool(T input, @list(int) ksize, @list(int) strides, @string padding, @string data_format)
    # T Y = MaxPool(T X, @AttrType.STRING auto_pad, @AttrType.INTS kernel_shape, @AttrType.INTS pads,
    #                   @AttrType.INTS strides)
    # above seems wrong - input[1] is ksize, input[2] is strides
    if len(node.input) < 3:
        kernel_shape = node.get_attr("ksize").ints
        kernel_shape = [kernel_shape[1], kernel_shape[2]]
        node.set_attr("kernel_shape", kernel_shape)
        strides = conv_dims_attr(node, "strides")
    else:
        kernel_shape = node.inputs[1].get_tensor_value()
        kernel_shape = [kernel_shape[1], kernel_shape[2]]
        node.set_attr("kernel_shape", kernel_shape)

        strides = node.inputs[2].get_tensor_value()
        strides = [strides[1], strides[2]]
        node.set_attr("strides", strides)

        ctx.remove_input(node, node.input[2])
        ctx.remove_input(node, node.input[1])

    conv_dims_attr(node, "dilations")

    add_padding(ctx, node, kernel_shape, strides)

    nodes = conv_convert_inputs(ctx, node, with_kernel=False)
    return nodes


def relu6_op(ctx, node, name, args):
    # relu6 = min(max(features, 0), 6)
    # since onnx does not have relu6, compose it with multiple ops.
    old_output = node.output[0]
    dtype = ctx.get_dtype(node.input[0])
    dtype = utils.ONNX_TO_NUMPY_DTYPE[dtype] if dtype else np.float32
    shape = ctx.get_shape(node.input[0])

    if -1 in shape:
        # if the shape has unknown dims we need to do something like this for opset < 8 (=no broadcast for min/max):
        # tz = sub(features, features)
        # t6 = add(6, tz)
        # relu6 = min(max(features, t0), t6)
        input_node = node.inputs[0]
        node.type = "Max"

        # const tensor 6
        six_name = utils.make_name(node.name)
        ctx.make_const(six_name, np.array([6.], dtype=dtype))

        # get a tensor of input shape with zeros
        sub_name = utils.make_name(input_node.name)
        sub_output = utils.port_name(sub_name)
        sub_node = Node(helper.make_node("Sub", [node.input[0], node.input[0]],
                                         [sub_output], name=sub_name), ctx)
        node.input.append(sub_output)

        # get a tensor of input shape with 6
        add_name = utils.make_name(input_node.name)
        add_output = utils.port_name(add_name)
        add_node = Node(helper.make_node("Add", [six_name, sub_output],
                                         [add_output], name=add_name), ctx)

        min_name = utils.make_name(node.name)
        min_node = ctx.insert_new_node_on_output("Min", node.output[0], name=min_name)
        min_node.input.append(add_output)
        ctx.copy_shape(old_output, min_node.output[0])
        return [sub_node, add_node, node, min_node]

    # if there is no unknown dim in shape we can use constants
    node.type = "Max"
    zero_name = utils.make_name(node.name)
    ctx.make_const(zero_name, np.zeros(shape, dtype=dtype))
    six_name = utils.make_name(node.name)
    six = np.zeros(shape, dtype=dtype)
    six.fill(6)
    ctx.make_const(six_name, six)
    node.input.append(zero_name)
    min_name = utils.make_name(node.name)
    min_node = ctx.insert_new_node_on_output("Min", node.output[0], name=min_name)
    min_node.input.append(six_name)
    ctx.copy_shape(old_output, min_node.output[0])
    return [node, min_node]


def relu6_op8(ctx, node, name, args):
    # relu6 = min(max(features, 0), 6) for opset >= 8
    # since onnx does not have relu6, compose it with multiple ops.
    old_output = node.output[0]
    dtype = ctx.get_dtype(node.input[0])
    dtype = utils.ONNX_TO_NUMPY_DTYPE[dtype] if dtype else np.float32
    node.type = "Max"

    # const tensor 6
    six_name = utils.make_name(node.name)
    ctx.make_const(six_name, np.array([6], dtype=dtype))
    zero_name = utils.make_name(node.name)
    ctx.make_const(zero_name, np.array([0], dtype=dtype))
    node.input.append(zero_name)
    min_name = utils.make_name(node.name)
    min_node = ctx.insert_new_node_on_output("Min", node.output[0], name=min_name)
    min_node.input.append(six_name)
    ctx.copy_shape(old_output, min_node.output[0])
    return [node, min_node]


def squareddifference_op(ctx, node, name, args):
    node.type = "Sub"
    op_name = utils.make_name(node.name)
    mul = ctx.insert_new_node_on_output("Mul", node.output[0], name=op_name)
    mul.input.append(node.output[0])
    return [node, mul]


def cast_op(ctx, node, name, args):
    # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
    # T2 output = Cast(T1 input, @STRING to)
    dst = node.get_attr("to")
    dst = tf2onnx.utils.ONNX_DTYPE_NAMES[dst]
    node.set_attr("to", dst)
    return node


def biasadd_op(ctx, node, name, args):
    # T output = BiasAdd(T value, T bias, @string data_format)
    # T output = BiasAddV1(T value, T bias)
    # TODO: for now use add. We may need to convert to NCHW.
    node.type = "Add"
    return broadcast_op(ctx, node, name, args)


def biasadd_op7(ctx, node, name, args):
    # T output = BiasAdd(T value, T bias, @string data_format)
    # T output = BiasAddV1(T value, T bias)
    # According TF bias_add definition, the input dim is always only 1.
    node.type = "Add"
    node = broadcast_op7(ctx, node, name, args)

    # on NHWC, bias will broadcast from largest dim, which is default onnx Add op broadcast behavior.
    if not node.is_nhwc():
        # however, in NCHW, bias should be at 2nd dim, which by default onnx Add op has no way to know,
        # so it needs being reshaped into 3-dim tensor before add
        shape0 = ctx.get_shape(node.input[0])
        shape1 = ctx.get_shape(node.input[1])
        if node.inputs[1].type == 'Const' and len(shape1) == 1:
            new_broadcast_shape = [shape1[0],] + [1,] * (len(shape0) - 2)
            shape_name = utils.make_name(node.name)
            ctx.make_const(shape_name, np.array(new_broadcast_shape, dtype=np.int64))
            op_name = node.input[1]
            reshape_node = ctx.insert_new_node_on_input(node, "Reshape", op_name)
            reshape_node.input.append(shape_name)
            ctx.set_shape(reshape_node.output[0], new_broadcast_shape)
            return [reshape_node, node]
    return node


def transpose_op(ctx, node, name, args):
    # T y = Transpose(T x, Tperm perm, @type Tperm)
    # T transposed = Transpose(T data, @INTS perm)
    if len(node.input) > 1:
        perm = node.inputs[1]
        if perm.is_const():
            # perms is passed as const
            dims = perm.get_tensor_value()
        else:
            # calculate perms from shape
            shape = ctx.get_shape(node.input[1])
            dims = [i for i in range(len(shape) - 1, -1)]
        ctx.remove_input(node, node.input[1])
        node.set_attr("perm", dims)
    else:
        # graph rewrite moved perm to attribute
        pass
    return node


def _wrap_concat_with_cast(ctx, node):
    """wrap concat in casts for opset < 8 since it only supports."""
    supported_types = [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16]
    dtype = node.dtype
    need_casting = dtype not in supported_types
    nodes = []
    if need_casting:
        output_name = node.output[0]
        # cast each inputs to float
        for i, inp in enumerate(node.inputs):
            input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[i])
            input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(input_cast.output[0], onnx_pb.TensorProto.FLOAT)
            nodes.append(input_cast)
        nodes.append(node)
        next_nodes = ctx.find_output_consumers(node.output[0])
        # cast output back to dtype unless the next op is a cast
        if next_nodes[0].type != "Cast":
            op_name = utils.make_name(node.name)
            output_cast = ctx.insert_new_node_on_output("Cast", output_name, name=op_name)
            output_cast.set_attr("to", dtype)
            output_cast.dtype = dtype
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(output_name, output_cast.output[0])
            nodes.append(output_cast)
    else:
        nodes.append(node)
    return nodes


def concat_op(ctx, node, name, args):
    # old concat op has axis as input[0]
    axis_node = node.inputs[0]
    axis = axis_node.get_tensor_value()
    ctx.remove_input(node, node.input[0])

    axis_val = axis[0]
    if axis_val < 0: # onnxruntime does not support -1 axis, but TF supports.
        input_shape = ctx.get_shape(node.input[0])
        axis_val = len(input_shape) + axis_val
    node.set_attr("axis", axis_val)

    if ctx.opset < 8:
        # opset < 8: might need to wrap concat in casts since only float is supported
        nodes = _wrap_concat_with_cast(ctx, node)
        return nodes
    return node


def concatv2_op(ctx, node, name, args):
    # T output = ConcatV2(T values, Tidx axis, @int N, @type Tidx)
    # T concat_result = Concat(T inputs, @INT axis)
    axis_node = node.inputs[-1]
    axis = axis_node.get_tensor_value()
    ctx.remove_input(node, node.input[-1])

    axis_val = axis[0]
    if axis_val < 0: # onnxruntime does not support -1 axis, but TF supports.
        input_shape = ctx.get_shape(node.input[0])
        axis_val = len(input_shape) + axis_val
    node.set_attr("axis", axis_val)

    if ctx.opset < 8:
        # opset < 8: might need to wrap concat in casts since only float is supported
        nodes = _wrap_concat_with_cast(ctx, node)
        return nodes
    return node


def slice_op(ctx, node, name, args):
    # T output = Slice(T input, Index begin, Index size, @type Index)
    # T output = Slice(T data, @INTS axes, @INTS ends, @INTS starts)
    starts = node.inputs[1].get_tensor_value()
    size = node.inputs[2].get_tensor_value()
    ends = np.add(starts, size)
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    node.set_attr("starts", starts)
    node.set_attr("ends", ends)
    return node


def gatherv2_op(ctx, node, name, args):
    # for GatherV2 axis come as input
    axis = node.inputs[2].get_tensor_value()
    ctx.remove_input(node, node.input[2])
    node.set_attr("axis", axis[0])
    return node


def split_op(ctx, node, name, args):
    # T output = Split(int32 split_dim, T value, @int num_split)
    # T outputs = Split(T input, @INT axis, @INTS split)
    split_dims = node.inputs[0].get_tensor_value()
    ctx.remove_input(node, node.input[0])
    node.set_attr("axis", split_dims[0])
    return node


def splitv_op(ctx, node, name, args):
    # T output = SplitV(T value, Tlen size_splits, int32 split_dim, @int num_split, @type Tlen)
    # T outputs = Split(T input, @INT axis, @INTS split)
    split = node.inputs[1].get_tensor_value()
    split_dims = node.inputs[2].get_tensor_value()
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    node.set_attr("split", split)
    node.set_attr("axis", split_dims[0])
    return node


def pad_op(ctx, node, name, args):
    # T output = Pad(T input, int32 paddings, @type Tpaddings), CONST model using default value
    #  or PadV2(T input, int32 paddings, T constant_value, @type Tpaddings), CONST mode - default value specified
    #  or MirrorPad(T input, int32 paddings, @type Tpaddings, @STRING mode), other mode.
    # T output = Pad(T data, @STRING mode, @INTS pads, @FLOAT value)
    paddings = np.array(node.inputs[1].get_tensor_value()).transpose().flatten()
    mode = node.get_attr("mode")
    if mode:
        mode = mode.s.decode("utf-8").lower()
        node.set_attr("mode", mode)
    if mode not in [None, "constant", "reflect"]:
        raise ValueError(mode + " pad mode is not supported")

    if mode in [None, "constant"] and len(node.input) == 3:
        const_val = node.inputs[2].get_tensor_value()[0]
        node.set_attr("value", const_val)
        ctx.remove_input(node, node.input[2])

    ctx.remove_input(node, node.input[1])
    node.set_attr("pads", paddings)
    return node


def rsqrt_op(ctx, node, name, args):
    node.type = "Sqrt"
    op_name = utils.make_name(node.name)
    reciprocal = ctx.insert_new_node_on_output("Reciprocal", node.output[0], name=op_name)
    ctx.copy_shape(node.output[0], reciprocal.output[0])
    return [node, reciprocal]


def expanddims_op(ctx, node, name, args):
    # T output = ExpandDims(T input, Tdim dim, @type Tdim)
    # tensorflow already infers the output shape so we can just take it
    shape = ctx.get_shape(node.output[0])
    node.type = "Reshape"
    ctx.remove_input(node, node.input[1])
    node.set_attr("shape", shape)
    return node


def expanddims_op7(ctx, node, name, args):
    shape = ctx.get_shape(node.output[0])
    shape_name = utils.make_name(node.name)
    ctx.make_const(shape_name, np.array(shape, dtype=np.int64))
    node.type = "Reshape"
    node.input[1] = shape_name
    return node


def stridedslice_op(ctx, node, name, args):
    # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
    not_supported_attr = ["ellipsis_mask", "new_axis_mask"]
    for attr_name in not_supported_attr:
        attr = node.get_attr(attr_name)
        if attr is not None and attr.i != 0:
            raise ValueError("StridedSlice: attribute " + attr_name + " not supported")

    begin = node.inputs[1].get_tensor_value()
    end = node.inputs[2].get_tensor_value()
    strides = node.inputs[3].get_tensor_value()
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask.i if end_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
    new_begin = []
    new_end = []
    axes = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    for idx, item in enumerate(begin):
        if strides[idx] != 1:
            raise ValueError("StridedSlice: only strides=1 is supported")
        axes.append(idx)
        mask = (shrink_axis_mask >> idx) & 1
        if mask != 0:
            new_begin.append(item)
            new_end.append(end[idx])
            needs_squeeze.append(idx)
            continue

        new_begin.append(item)
        mask = (end_mask >> idx) & 1
        if mask != 0:
            new_end.append(sys.maxsize)
        else:
            new_end.append(end[idx])

    node.set_attr("starts", new_begin)
    node.set_attr("ends", new_end)
    node.set_attr("axes", axes)
    node.type = "Slice"
    ctx.remove_input(node, node.input[3])
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    nodes = [node]
    if needs_squeeze:
        name = utils.make_name(node.name)
        squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
        squeeze_node.set_attr("axes", needs_squeeze)
        nodes.append(squeeze_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(squeeze_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], squeeze_node.output[0])

    # onnx slice as of opset 7 does only take float tensors ... cast if needed
    input_dtype = ctx.get_dtype(node.input[0])
    if input_dtype in [onnx_pb.TensorProto.INT32, onnx_pb.TensorProto.INT64]:
        cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.input[0], cast_node.output[0])
        nodes.insert(0, cast_node)
        name = utils.make_name(node.name)
        cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
        cast_node.set_attr("to", input_dtype)
        ctx.set_dtype(cast_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], cast_node.output[0])
        nodes.append(cast_node)
    return nodes


def pow_op(ctx, node, name, args):
    if ctx.is_target(TARGET_CAFFE2):
        # workaround a bug in caffe2 pre Feb2018, pow(a, b) becomes np.exp(np.log(a) * b)
        node.type = "Log"
        b = node.input[1]
        ctx.remove_input(node, node.input[1])
        op_name = utils.make_name(node.name)
        mul_op = ctx.insert_new_node_on_output("Mul", node.output[0], name=op_name)
        mul_op.input.append(b)
        op_name = utils.make_name(node.name)
        exp_op = ctx.insert_new_node_on_output("Exp", mul_op.output[0], name=op_name)
        ctx.copy_shape(node.output[0], exp_op.output[0])
        return [node, broadcast_op(ctx, mul_op, name, args), exp_op]
    return node


def lrn_op(ctx, node, name, args):
    # FIXME: numerical results are not correct
    # ONNX: Each input value is divided by (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
    # TF: sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    #     output = input / (bias + alpha * sqr_sum) ** beta
    depth_radius = node.get_attr("depth_radius")
    if depth_radius:
        size = depth_radius.i
    else:
        size = 5
    node.set_attr("size", size)
    return node


def upsample_op7(ctx, node, name, args):
    mode = args[0]
    shape = ctx.get_shape(node.input[0])
    target_shape = node.inputs[1].get_tensor_value()
    # https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest_neighbor
    # wants the input to be NHWC - adjust target_shape to this.
    n, h, w, c = shape
    nh, nw = target_shape
    # scaler is nchw
    scaler = [1., 1., float(nh) / h, float(nw) / w]
    node.set_attr("scales", scaler)
    node.set_attr("mode", mode)
    ctx.remove_input(node, node.input[1])
    node.data_format = "NHWC"
    nodes = conv_convert_inputs(ctx, node, with_kernel=False)
    return nodes


def multinomial_op(ctx, node, name, args):
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
    node.set_attr("sample_size", sample_size[0])
    ctx.remove_input(node, node.input[1])
    return node


def topk_op(ctx, node, name, args):
    k = node.inputs[1].get_tensor_value()
    node.set_attr("k", k[0])
    node.type = "TopK"
    ctx.remove_input(node, node.input[1])

    # the second of TopK operator must be INT64 per ONNX requires
    ctx.override_dtype(port_name(name, 1), onnx_pb.TensorProto.INT64)
    return node


def tile_op7(ctx, node, name, args):
    # onnx wants shape input to be int64
    return _convert_shapenode_to_int64(ctx, node, 1)


def reorganize_data_op(ctx, node, name, args):
    block_size = node.get_attr("block_size")
    node.set_attr("blocksize", block_size.i)
    nodes = conv_convert_inputs(ctx, node, with_kernel=False)
    return nodes


def minmax_op(ctx, node, name, args):
    # tensorflow minimum/maximum does support broadcast, onnx < opset 8 does not.
    # handle this by doing something like:
    # y = min(x1, add(x2, sub(x1, x1))), where x1, x2 are the inputs and x2 is a scalar
    # this will create a tensor of zeros of the shape of x1, adds x2 to it (which broadcasts) and use that for min.
    shapeo = ctx.get_shape(node.output[0])
    needs_broadcast_op = []
    has_correct_shape = []
    if ctx.opset < 8:
        for i, input_name in enumerate(node.input):
            if ctx.get_shape(input_name) != shapeo:
                needs_broadcast_op.append(i)
            else:
                has_correct_shape.append(input_name)
    if needs_broadcast_op:
        new_nodes = []
        has_correct_shape = has_correct_shape[0]
        for i in needs_broadcast_op:
            input_node = node.inputs[i]
            sub_name = utils.make_name(input_node.name)
            sub_output = utils.port_name(sub_name)
            # get a tensor with zeros (since there is no Fill op as of opset8)
            sub_node = Node(helper.make_node("Sub", [has_correct_shape, has_correct_shape],
                                             [sub_output], name=sub_name), ctx)
            add_name = utils.make_name(input_node.name)
            add_output = utils.port_name(add_name)
            # use add as 'broadcast' op
            add_node = Node(helper.make_node("Add", [input_node.output[0], sub_output],
                                             [add_output], name=add_name), ctx)
            node.input[i] = add_output
            new_nodes.append(sub_node)
            new_nodes.append(add_node)
        new_nodes.append(node)
        return new_nodes

    return node


def pack_op(ctx, node, name, args):
    # hack to make up for the missing onnx pack op
    axis = node.get_attr("axis").i
    nodes = []
    inputs = []
    dtype = None
    # insert Unsqueeze on each input
    for i, n in enumerate(node.inputs):
        op_name = utils.make_name(node.name)
        output_name = port_name(op_name)
        dtype = ctx.get_dtype(node.input[i])
        shape = ctx.get_shape(node.input[i])
        new_node = Node(helper.make_node("Unsqueeze", [node.input[i]], [output_name], name=op_name, axes=[axis]), ctx)
        node.input[i] = output_name
        ctx.set_dtype(output_name, dtype)
        ctx.set_shape(output_name, shape)
        nodes.append(new_node)
        inputs.append(output_name)
    # concat all unqueezes
    op_name = utils.make_name(node.name)
    output_name = port_name(op_name)
    concat = Node(helper.make_node("Concat", inputs, [output_name], name=op_name, axis=axis), ctx)
    ctx.copy_shape(node.output[0], concat.output[0])
    ctx.set_dtype(concat.output[0], dtype)
    ctx.replace_all_inputs(ctx.get_nodes(), node.output[0], output_name)
    return [concat] + nodes


def unpack_op(ctx, node, name, args):
    # hack to make up for the missing onnx unpack op
    axis = node.get_attr("axis").i
    # split the tensor into n outputs
    node.type = "Split"
    nodes = [node]
    # for each output we need to squeeze axis
    for i, n in enumerate(node.output):
        op_name = utils.make_name(node.name)
        output_name = port_name(op_name, i)
        dtype = ctx.get_dtype(n)
        new_node = Node(helper.make_node("Squeeze", [n], [output_name], name=op_name, axes=[axis]), ctx)
        ctx.set_dtype(output_name, dtype)
        ctx.copy_shape(n, output_name)
        nodes.append(new_node)
        ctx.replace_all_inputs(ctx.get_nodes(), n, output_name)
    return nodes


def onehot_op(ctx, node, name, args):
    # until there is no onehot op in onnx, a workaround using gather from eye
    indices_name = node.input[0]
    indices_shape = ctx.get_shape(indices_name)
    if len(indices_shape) != 1:
        # TODO: this works for rank=1 but tensorflow supports more than this.
        # Same principle should work but we need to implemtn our own eye.
        raise ValueError("onehot op: only rank1 is supported")
    axis = node.get_attr("axis")
    # axis becomes axis for gather
    node.set_attr("axis", 0)
    depth = node.inputs[1].get_tensor_value()[0]
    on = node.inputs[2].get_tensor_value()[0]
    off = node.inputs[3].get_tensor_value()[0]
    dtype = node.inputs[2].get_tensor_type()
    eye = np.eye(depth, dtype=dtype)
    if on != 0:
        eye[eye == 1] = on
        eye[eye == 0] = off
    else:
        eye[eye == 0] = off
        eye[eye == 1] = on
    const_name = utils.make_name(node.name)
    ctx.make_const(const_name, eye)
    # setup gather inputs
    del node.input[:]
    node.input.append(const_name)
    node.input.append(indices_name)
    node.type = "Gather"
    if axis.i == 0:
        # TODO: revisit for rank > 1
        name = utils.make_name(node.name)
        transpose_node = ctx.insert_new_node_on_output("Transpose", node.output[0], name)
        ctx.copy_shape(node.output[0], transpose_node.output[0])
        return [node, transpose_node]
    return node


def fused_batchnorm_op7(ctx, node, name, args):
    node.type = "BatchNormalization"
    # tf inputs: x, scale, bias, mean, variance
    # tf outputs: y, batch_mean, batch_var
    # a: data_format, epsilon, is_training
    # onnx inputs: X, scale, B, mean, variance, attributes: epsilon, momentum=0.9, spatial : 1
    # output: y, mean, var, savedmean, savedvar,
    nodes = conv_convert_inputs(ctx, node, with_kernel=False)
    scale_shape = ctx.get_shape(node.input[1])
    mean_shape = ctx.get_shape(node.input[3])
    var_shape = ctx.get_shape(node.input[4])
    val_type = utils.ONNX_TO_NUMPY_DTYPE[node.inputs[1].dtype]

    if mean_shape != scale_shape:
        new_mean_value = np.array(np.resize(node.inputs[3].get_tensor_value(), scale_shape), dtype=val_type)
        new_mean_node_name = utils.make_name(node.name)
        ctx.make_const(new_mean_node_name, new_mean_value)
        node.input[3] = new_mean_node_name

    if var_shape != scale_shape:
        new_var_value = np.array(np.resize(node.inputs[4].get_tensor_value(), scale_shape), dtype=val_type)
        new_val_node_name = utils.make_name(node.name)
        ctx.make_const(new_val_node_name, new_var_value)
        node.input[4] = new_val_node_name

    return nodes


def matmul_op(ctx, node, name, args):
    # tensorflow allows transpose and conjugated. If found, insert the required transpose.
    # We could use Gemm as well but tensorflow does not pass bias in matmul.
    attrs = ["transpose_a", "transpose_b", "adjoint_a", "adjoint_b", "adj_x", "adj_y"]
    attrs_val = [node.get_attr(attr) for attr in attrs]
    attrs_val = [0 if val is None else val.i for val in attrs_val]

    if any(attrs_val[2:]):
        # conjugation operation on complex data not supported in onnx for now, so if it's complex than raise exception
        if node.dtype not in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.DOUBLE]:
            raise ValueError("dtype " + node.dtype + " is not supported in onnx matmul for now")

    transpose_a = (attrs_val[0] + attrs_val[2] + attrs_val[4]) % 2
    transpose_b = (attrs_val[1] + attrs_val[3] + attrs_val[5]) % 2

    nodes = []
    if transpose_a != 0:
        transpose = ctx.insert_new_node_on_input(node, "Transpose", node.input[0])
        nodes.insert(0, transpose)
    if transpose_b != 0:
        transpose = ctx.insert_new_node_on_input(node, "Transpose", node.input[1])
        nodes.insert(0, transpose)

    unsupported = ["a_is_sparse", "b_is_sparse"]
    for i in unsupported:
        val = node.get_attr(i)
        if val is not None and val.i != 0:
            raise ValueError(node.type + " attribute " + i + " is not supported")
    nodes.append(node)
    return nodes

def fill_op7(ctx, node, name, args):
    # T output = Fill(int32 dims, T value, @int32 index_type)
    # T outputs = Tile(T value, int64 repeats (e.g. dims))
    nodes = [node]
    fill_shape = ctx.get_shape(node.input[0])
    fill_shape_dims = fill_shape[0]
    val_dtype = ctx.get_dtype(node.input[1])
    val_shape = ctx.get_shape(node.input[1])

    need_cast = val_dtype != onnx_pb.TensorProto.FLOAT and ctx.opset < 8
    new_dtype = val_dtype
    if need_cast:
        new_dtype = onnx_pb.TensorProto.FLOAT
        attr = {"to": new_dtype}
        cast_to_float = ctx.insert_new_node_on_input(node, "Cast", node.input[1], name=None, **attr)
        nodes.insert(0, cast_to_float)
        ctx.set_dtype(cast_to_float.output[0], new_dtype)
        ctx.set_shape(cast_to_float.output[0], val_shape)

    for i in range(fill_shape_dims):
        attr = {"axes": [0]}
        shape = ctx.get_shape(node.input[1])
        unsqueeze_node = ctx.insert_new_node_on_input(node, "Unsqueeze", node.input[1], name=None, **attr)
        nodes.insert(0, unsqueeze_node)
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
    nodes.insert(0, tile_shape_int64)

    tmp = node.input[0]
    node.input[0] = node.input[1]
    node.input[1] = tmp
    node.type = "Tile"
    ctx.set_dtype(node.output[0], new_dtype)

    if need_cast:
        attr = {"to": val_dtype}
        op_name = utils.make_name(node.name)
        cast_back = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name, **attr)
        nodes.insert(0, cast_back)
        ctx.set_dtype(cast_back.output[0], val_dtype)

    return nodes

def fill_op(ctx, node, name, args):
    node.type = "ConstantLike"
    # both shape and value in tensorflow are passed as tensor.
    # In onnx the value is an attribute so we need to fetch the value as const which
    # sooner or later will be a problem for tensorflow-onnx.
    shape = ctx.get_shape(node.output[0])
    value = node.inputs[1].get_tensor_value()
    value_proto = numpy_helper.from_array(node.inputs[1].get_tensor())
    dtype = value_proto.data_type
    # onnx spec says value MUST be float.
    node.set_attr("value", float(value[0]))
    node.set_attr("shape", shape)
    node.set_attr("dtype", dtype)
    del node.input[:]
    return node


# map tensorflow ops to onnx ops. The format below is
# "TFOP": func_to_map, ["OnnxOp", ...]
#
_OPSET_4 = {
    "Abs": (direct_op, []),
    "Add": (broadcast_op, []),
    "ArgMax": (arg_minmax_op, []),
    "ArgMin": (arg_minmax_op, []),
    "AvgPool": (pool_op, ["AveragePool"]),
    "AvgPool3D": (pool_op, ["AveragePool"]),
    "BiasAdd": (biasadd_op, []),
    "BiasAddV1": (biasadd_op, []),
    "Cast": (cast_op, []),
    "Concat": (concat_op, ["Concat"]),
    "ConcatV2": (concatv2_op, ["Concat"]),
    "Const": (const_op, []),
    "ConstV2": (const_op, []),
    "Conv2D": (conv_op, ["Conv"]),
    "Conv2DBackpropInput": (convtranspose_op, ["ConvTranspose"]),
    "Conv3D": (conv_op, ["Conv"]),
    "Equal": (broadcast_op, []),
    "ExpandDims": (expanddims_op, []),
    "DepthwiseConv2d": (depthwiseconv_op, ["Conv"]),
    "DepthwiseConv2dNative": (depthwiseconv_op, ["Conv"]),
    "Dropout": (direct_op, []),
    "Elu": (direct_op, []),
    "Exp": (direct_op, []),
    "Floor": (direct_op, []),
    "Flatten": (direct_op, []),
    "Gather": (direct_op, ["Gather"]),
    "GatherV2": (gatherv2_op, ["Gather"]),
    "Greater": (broadcast_op, []),
    "Identity": (identity_op, ["Identity"]),
    "Less": (broadcast_op, []),
    "Log": (direct_op, []),
    "LRN": (lrn_op, []),
    "LogicalAnd": (broadcast_op, ["And"]),
    "LogicalOr": (broadcast_op, ["Or"]),
    "Max": (reduce_op, ["ReduceMax"]),
    "MatMul": (matmul_op, ["MatMul"]),
    "BatchMatMul": (matmul_op, ["MatMul"]),
    "Maximum": (minmax_op, ["Max"]),
    "MaxPool": (pool_op, ["MaxPool"]),
    "MaxPoolV2": (pool_op, ["MaxPool"]),
    "Mean": (reduce_op, ["ReduceMean"]),
    "Min": (reduce_op, ["ReduceMin"]),
    "Minimum": (minmax_op, ["Min"]),
    "MirrorPad": (pad_op, ["Pad"]),
    "Mul": (broadcast_op, []),
    "Neg": (direct_op, []),
    "NoOp": (no_op, []),
    "NotEqual": (direct_op, ["Not"]),
    "Pad": (pad_op, []),
    "PadV2": (pad_op, ["Pad"]),
    "Placeholder": (placeholder_op, []),
    "PlaceholderV2": (placeholder_op, []),
    "PlaceholderWithDefault": (placeholder_op, []),
    "Pow": (pow_op, []),
    "Prod": (reduce_op, ["ReduceProd"]),
    "RandomNormal": (direct_op, []),
    "RandomUniform": (direct_op, []),
    "RandomNormalLike": (direct_op, []),
    "RandomUniformLike": (direct_op, []),
    "RealDiv": (broadcast_op, ["Div"]),
    "Reciprocal": (direct_op, []),
    "Relu": (direct_op, ["Relu"]),
    "Relu6": (relu6_op, []),
    "Reshape": (reshape_op, ["Reshape"]),
    "Rsqrt": (rsqrt_op, []),
    "Shape": (direct_op, []),
    "Size": (direct_op, []),
    "Sigmoid": (direct_op, []),
    "Slice": (slice_op, []),
    "SplitV": (splitv_op, ["Split"]),
    "Split": (split_op, ["Split"]),
    "Squeeze": (squeeze_op, []),
    "Sqrt": (direct_op, []),
    "Square": (square_op, []),
    "SquaredDifference": (squareddifference_op, []),
    "Softmax": (direct_op, ["Softmax"]),
    "StopGradient": (identity_op, ["Identity"]),
    "StridedSlice": (stridedslice_op, []),
    "Sub": (broadcast_op, []),
    "Sum": (reduce_op, ["ReduceSum"]),
    "Tanh": (direct_op, []),
    "Transpose": (transpose_op, []),
    "TopKV2": (topk_op, []),
    "SpaceToDepth": (reorganize_data_op, []),
    "DepthToSpace": (reorganize_data_op, []),
    "Pack": (pack_op, []),
    "Unpack": (unpack_op, []),
}

_OPSET_5 = {
    "Reshape": (reshape_op5, []),
    "ExpandDims": (expanddims_op7, []),
    "OneHot": (onehot_op, []),
}

_OPSET_6 = {
    "AddN": (direct_op, ["Sum"]),
}

_OPSET_7 = {
    "Tile": (tile_op7, []),
    "ResizeNearestNeighbor": (upsample_op7, ["Upsample", "nearest"]),
    "ResizeBilinear": (upsample_op7, ["Upsample", "linear"]),
    "BiasAdd": (biasadd_op7, []),
    "BiasAddV1": (biasadd_op7, []),
    "Add": (broadcast_op7, []),
    "Sub": (broadcast_op7, []),
    "Mul": (broadcast_op7, []),
    "RealDiv": (broadcast_op7, ["Div"]),
    "Div": (broadcast_op7, ["Div"]),
    "TruncateDiv": (broadcast_op7, ["Div"]),
    "LogicalAnd": (broadcast_op7, ["And"]),
    "LogicalOr": (broadcast_op7, ["Or"]),
    "Greater": (broadcast_op7, []),
    "Less": (broadcast_op7, []),
    "Pow": (direct_op, []),
    "Cast": (direct_op, []),
    "Acos": (direct_op, []),
    "Asin": (direct_op, []),
    "Atan": (direct_op, []),
    "Cos": (direct_op, []),
    "Sin": (direct_op, []),
    "Tan": (direct_op, []),
    "Multinomial": (multinomial_op, []),
    "Fill": (fill_op7, []),
    "FusedBatchNorm": (fused_batchnorm_op7, []),
    "FusedBatchNormV2": (fused_batchnorm_op7, []),
}

_OPSET_8 = {
    "Relu6": (relu6_op8, []),  # make use of min/max broadcast
}

_OPSET_9 = {
    "Fill": (fill_op, []),
}

_OPSETS = [
    (4, _OPSET_4),
    (5, _OPSET_5),
    (6, _OPSET_6),
    (7, _OPSET_7),
    (8, _OPSET_8),
    (9, _OPSET_9),
]


def rewrite_random_uniform(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', inputs=[
                OpTypePattern('RandomUniform', name='input1', inputs=["*"]),
                OpTypePattern('Sub', name='input2', inputs=["*", "*"]),
            ]), None
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        input2 = match.get_op('input2')
        output = match.get_op('output')
        # max is on input 0
        tmax = input2.inputs[0].get_tensor_value()[0]
        tmin = input2.inputs[1].get_tensor_value()[0]
        dtype = output.dtype
        op_name = utils.make_name("RandomUniform")
        out_name = port_name(op_name)
        ru_op = match.get_op('input1')
        if ru_op.inputs[0].type == "Shape":
            shape_op = ru_op.inputs[0]
            new_node = Node(helper.make_node("RandomUniformLike",
                                             [shape_op.input[0]], [out_name], name=op_name,
                                             low=tmin, high=tmax, dtype=dtype), g)
        else:
            shape = g.get_shape(output.output[0])
            new_node = Node(helper.make_node("RandomUniform",
                                             [], [out_name], name=op_name,
                                             low=tmin, high=tmax, dtype=dtype, shape=shape), g)
        ops = g.replace_subgraph(ops, match, [], [output], [], [new_node])

    return ops


def rewrite_transpose(g, ops):
    pattern = \
        OpTypePattern('Transpose', name='output', inputs=[
            OpTypePattern(None),
            OpTypePattern('Sub', inputs=[
                OpTypePattern('Sub', inputs=["*", "*"]),
                OpTypePattern('Range', inputs=["*", "*", "*"]),
            ]),
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        output = match.get_op('output')
        shape = g.get_shape(output.input[0])
        dims = [i for i in range(len(shape) - 1, -1, -1)]
        output.set_attr("perm", dims)
        g.remove_input(output, output.input[1])
        ops = g.replace_subgraph(ops, match, [], [], [], [])
        ops.append(output)
    return ops


def rewrite_random_normal(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', name='input2', inputs=[
                OpTypePattern('RandomStandardNormal', name='input1', inputs=["*"]), "*"
            ]), "*"
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        output = match.get_op('output')
        mean = output.inputs[1].get_tensor_value()[0]
        dtype = output.dtype
        op_name = utils.make_name("RandomNormal")
        out_name = port_name(op_name)

        rn_op = match.get_op('input1')
        if rn_op.inputs[0].type == "Shape":
            shape_op = rn_op.inputs[0]
            new_node = Node(helper.make_node("RandomNormalLike", [shape_op.input[0]], [out_name],
                                             name=op_name, mean=mean, scale=1.0,
                                             dtype=dtype), g)
            ops = g.replace_subgraph(ops, match, [], [output], [], [new_node])
        else:
            shape = g.get_shape(output.output[0])
            new_node = Node(helper.make_node("RandomNormal", [], [out_name],
                                             name=op_name, shape=shape, mean=mean, scale=1.0,
                                             dtype=dtype), g)
            ops = g.replace_subgraph(ops, match, [], [output], [], [new_node])

    return ops


def rewrite_dropout(g, ops):
    pattern = \
        OpTypePattern('Mul', name='outputs', inputs=[
            OpTypePattern('RealDiv', name="input2"),
            OpTypePattern('Floor', inputs=[
                OpTypePattern('Add', inputs=[
                    OpTypePattern(None, name="input3"),
                    OpTypePattern('RandomUniform'),
                ])
            ]),
        ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        inputs2 = match.get_op('input2')
        outputs = match.get_op('outputs')
        op_name = utils.make_name("Dropout")
        out_name = port_name(op_name)
        new_node = Node(helper.make_node("Dropout", [inputs2.input[0]], [out_name], name=op_name, ratio=1.0), g)
        ops = g.replace_subgraph(ops, match, [inputs2], [outputs], [new_node], [new_node])

    return ops


def rewrite_flatten(g, ops):
    pattern = \
        OpTypePattern('Reshape', name='outputs', inputs=[
            OpTypePattern("*", name="input2"),
            OpTypePattern('Pack', inputs=[
                OpTypePattern('StridedSlice', inputs=[
                    "*", "*", "*", "*",
                ]),
                "*",
            ]),
        ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        inputs2 = match.get_op('input2')
        outputs = match.get_op('outputs')
        op_name = utils.make_name("Flatten")
        out_name = port_name(op_name)
        new_node = Node(helper.make_node("Flatten", [inputs2.output[0]], [out_name], name=op_name), g)
        g.replace_all_inputs(ops, outputs.output[0], out_name)
        to_be_removed = [node for node in match.get_nodes() if node != inputs2]
        for i in range(len(ops) - 1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
        ops.append(new_node)
    return ops


def rewrite_incomplete_type_support(g, ops):
    """
    for ops that have inclomplete type support, insert casts.
    This is needed for some tensor ops in opset7 and for some ops in winml-rs5.
    It is not helping performance but better than the model not working at all.
    """
    new_ops = []
    needs_pass2 = False
    for op in ops:
        if op.type in ["Unsqueeze", "Mul", "Concat"]:
            cast_inserted = []
            output_dtype = None
            # insert casts on inputs if the runtime only supports float
            for i, node in enumerate(op.inputs):
                input_name = op.input[i]
                dtype = g.get_dtype(input_name)
                if dtype != onnx_pb.TensorProto.FLOAT:
                    output_dtype = dtype
                    if node.type == "Cast":
                        node.set_attr("to", onnx_pb.TensorProto.FLOAT)
                        g.set_dtype(input_name, onnx_pb.TensorProto.FLOAT)
                    else:
                        cast_node = g.insert_new_node_on_input(op, "Cast", input_name)
                        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
                        g.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
                        g.copy_shape(input_name, cast_node.output[0])
                        cast_inserted.append(cast_node)
            if output_dtype:
                # insert reverse cast if needed
                for output_name in op.output:
                    name = utils.make_name(op.name)
                    output_cast = g.insert_new_node_on_output("Cast", output_name, name=name)
                    output_cast.set_attr("to", output_dtype)
                    g.set_dtype(output_cast.output[0], output_dtype)
                    g.copy_shape(output_name, output_cast.output[0])
                    cast_inserted.append(output_cast)
            if cast_inserted:
                new_ops.extend(cast_inserted)
                needs_pass2 = True
        new_ops.append(op)

    if needs_pass2:
        # possible that we inserted casts that don't annything ... get rid of them
        ops = new_ops
        new_ops = []
        for op in ops:
            if op.type != "Cast":
                new_ops.append(op)
                continue
            dtype_in = g.get_dtype(op.input[0])
            dtype_out = g.get_dtype(op.output[0])
            if dtype_in != dtype_out:
                new_ops.append(op)
                continue

            # input and output have the same dtype ... remove the cast
            output_name = op.output[0]
            for node in ops:
                for i, input_name in enumerate(node.input):
                    if output_name == input_name:
                        node.input[i] = op.input[0]

    return new_ops


def tensorflow_onnx_mapping(g, continue_on_error, custom_op_handlers):
    mapped_op = collections.Counter()
    unmapped_op = collections.Counter()

    # create ops mapping for the desired opset
    ops_mapping = {}
    for target_opset, op_map in _OPSETS:
        if target_opset <= g.opset:
            ops_mapping.update(op_map)

    # apply custom ops on top of the assembled opset. We can either completment the opset
    # or override existing ops with a custom op.
    if custom_op_handlers is not None:
        custom_opset = {k: [v, []] for k, v in custom_op_handlers.items()}
        ops_mapping.update(custom_opset)

    ops = g.get_nodes()
    onnx_nodes = []
    for node in ops:
        if node.need_skip():
            log.debug("explictly skip node " + node.name)
            onnx_nodes.append(node)
            continue
        op = node.type
        map_info = ops_mapping.get(op)
        if map_info is None:
            if continue_on_error:
                unmapped_op[op] += 1
                onnx_nodes.append(node)
                continue
            else:
                raise ValueError("tensorflow op " + op + " is not supported")
        mapped_op[op] += 1
        func, args = map_info
        onnx_node = None
        if args:
            node.type = args[0]
            args = args[1:]
        try:
            onnx_node = func(g, node, node.name, args)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            log.error("node %s: exception %s" % (node.name, ex))
            ex_ext = traceback.format_exception(type_, value_, traceback_)
            if continue_on_error:
                log.info(ex_ext)
                onnx_nodes.append(node)
            else:
                raise ex
        if onnx_node:
            if isinstance(onnx_node, list):
                onnx_nodes.extend(onnx_node)
            else:
                onnx_nodes.append(onnx_node)

    g.set_nodes(onnx_nodes)

    return mapped_op, unmapped_op


def transpose_inputs(ctx, inputs_as_nchw):
    """Insert a transpose from NHWC to NCHW on model input on users request."""
    ops = []
    for node in ctx.get_nodes():
        for idx, output_name in enumerate(node.output):
            if output_name in inputs_as_nchw:
                shape = ctx.get_shape(output_name)
                if len(shape) != len(NCHW_TO_NHWC):
                    log.warning("transpose_input for %s: shape must be rank 4, ignored" % output_name)
                    ops.append(node)
                    continue
                # insert transpose
                op_name = utils.make_name(node.name)
                transpose = ctx.insert_new_node_on_output("Transpose", output_name, name=op_name)
                transpose.set_attr("perm", NCHW_TO_NHWC)
                transpose.inserted_nchw = True
                ctx.copy_shape(output_name, transpose.output[0])
                ctx.set_shape(output_name, np.array(shape)[NHWC_TO_NCHW])
                ops.append(transpose)
                ops.append(node)
                continue
            ops.append(node)
    ctx.set_nodes(ops)


def tf_optimize(inputs, outputs, graph_def, fold_constant=None):
    """Optimize tensorflow graph for inference."""
    transforms = []
    if fold_constant:
        transforms.extend([
            "fold_constants(ignore_errors=true)",
            "remove_attribute(attribute_name=_class)", # remove node colocation attributes
        ])

    transforms.extend([
        "fold_batch_norms",
        "fold_old_batch_norms",
    ])
    needed_names = [utils.node_name(i) for i in inputs] + [utils.node_name(i) for i in outputs]
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)
    graph_def = TransformGraph(graph_def, inputs, outputs, transforms)
    return graph_def


def process_tf_graph(tf_graph, continue_on_error=False, verbose=False, target=None,
                     opset=None, custom_op_handlers=None, custom_rewriter=None,
                     extra_opset=None, shape_override=None, inputs_as_nchw=None):
    """Convert tensorflow graph to onnx graph.
        Args:
            tf_graph: tensorflow graph
            continue_on_error: if an op can't be processed (aka there is no mapping), continue
            verbose: print summary stats
            target: list of workarounds applied to help certain platforms
            opset: the opset to be used (int, default is latest)
            custom_op_handlers: dictionary of custom ops handlers
            custom_rewriter: list of custom graph rewriters
            extra_opset: list of extra opset's, for example the opset's used by custom ops
            shape_override: dict with inputs that override the shapes given by tensorflow
            inputs_as_nchw: transpose inputs in list from nchw to nchw
        Return:
            onnx graph
    """
    def topological_sort(ops):
        if not continue_on_error:
            g.topological_sort(ops)
        else:
            try:
                g.topological_sort(ops)
            except:  # pylint: disable=bare-except
                # if we continue on error, ignore graph cycles so we can report all missing ops
                pass

    if shape_override is None:
        shape_override = {}
    if inputs_as_nchw is None:
        inputs_as_nchw = []
    if target is None:
        target = DEFAULT_TARGET

    onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes = tensorflow_to_onnx(tf_graph, shape_override)

    g = Graph(onnx_nodes, output_shapes, dtypes, target, opset, extra_opset)
    if inputs_as_nchw:
        transpose_inputs(g, inputs_as_nchw)

    # pre-processing graph rewrites
    # bi-directional re-writer should be place after single directional re-writer
    rewriters = [rewrite_transpose, rewrite_flatten, rewrite_random_uniform,
                 rewrite_random_normal, rewrite_dropout,
                 rewrite_single_direction_lstm, rewrite_bi_direction_lstm,
                 rewrite_single_direction_gru, rewrite_single_direction_grublock,
                 rewrite_bi_direction_gru]

    if custom_rewriter is not None:
        rewriters.extend(custom_rewriter)

    ops = g.get_nodes()
    for rewrite in rewriters:
        ops = rewrite(g, ops)
        g.set_nodes(ops)
    topological_sort(g.get_nodes())

    if custom_op_handlers is None:
        custom_op_handlers = {}
    mapped_op, unmapped_op = tensorflow_onnx_mapping(g, continue_on_error, custom_op_handlers)

    # post-processing rewriters
    late_rewriters = []
    if TARGET_RS5 in target:
        late_rewriters.append(rewrite_incomplete_type_support)
    if late_rewriters:
        topological_sort(g.get_nodes())
        ops = g.get_nodes()
        for rewrite in late_rewriters:
            ops = rewrite(g, ops)
            g.set_nodes(ops)

    # onnx requires topological sorting
    topological_sort(g.get_nodes())

    g.update_proto()

    if verbose:
        print("tensorflow ops: {}".format(op_cnt))
        print("tensorflow attr: {}".format(attr_cnt))
        print("onnx mapped: {}".format(mapped_op))
        print("onnx unmapped: {}".format(unmapped_op))

    return g
