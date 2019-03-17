# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - rewrite tensorflow graph to onnx graph
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
from tf2onnx.function import *  # pylint: disable=wildcard-import
from tf2onnx.graph import Graph
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.rewriter.cond_rewriter import rewrite_cond
from tf2onnx.rewriter.random_uniform import rewrite_random_uniform, rewrite_random_uniform_fold_const
from tf2onnx.rewriter.leakyrelu_rewriter import rewrite_leakyrelu
from tf2onnx.rewriter.rnn import rewrite_bi_direction_gru
from tf2onnx.rewriter.rnn import rewrite_custom_rnn_cell
from tf2onnx.rewriter.rnn import rewrite_generic_loop
from tf2onnx.rewriter.rnn import rewrite_single_direction_gru
from tf2onnx.rewriter.rnn import rewrite_single_direction_grublock
from tf2onnx.rewriter.rnn import rewrite_single_direction_lstm, rewrite_bi_direction_lstm
from tf2onnx.shape_inference import infer_shape_for_graph, set_shape_from_inputs_broadcast
from tf2onnx.utils import port_name

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx")

# Target for the generated onnx graph. It possible targets:
# onnx-1.1 = onnx at v1.1 (winml in rs4 is based on this)
# caffe2 = include some workarounds for caffe2 and winml
TARGET_RS4 = "rs4"
TARGET_RS5 = "rs5"
TARGET_RS6 = "rs6"
TARGET_CAFFE2 = "caffe2"
POSSIBLE_TARGETS = [TARGET_RS4, TARGET_RS5, TARGET_RS6, TARGET_CAFFE2]
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
    ignored_attr = ["unknown_rank", "_class", "Tshape", "use_cudnn_on_gpu", "Index", "Tpaddings",
                    "TI", "Tparams", "Tindices", "Tlen", "Tdim", "dynamic_size", "Tmultiples",
                    "Tblock_shape", "Tcrops", "index_type", "Taxis", "U", "maxval",
                    "Tout", "Tlabels", "Tindex", "element_shape"]
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
                    shape = None
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
                attr[a] = utils.map_tf_dtype(utils.get_tf_node_attr(node, "dtype"))
            elif a == "T":
                dtype = utils.get_tf_node_attr(node, "T")
                if dtype:
                    if not isinstance(dtype, list):
                        dtypes[node.name] = utils.map_tf_dtype(dtype)
            elif a in ["output_type", "output_dtype", "out_type", "Tidx", "out_idx"]:
                # Tidx is used by Range
                # out_idx is used by ListDiff
                attr[a] = utils.map_tf_dtype(utils.get_tf_node_attr(node, a))
            elif a == "shape":
                attr[a] = utils.get_shape(node)
            elif a == "Tperm":
                pass
            elif a == "_output_shapes":
                attr[a] = utils.get_shape(node)
            elif a == "value":
                onnx_tensor = utils.tf_to_onnx_tensor(utils.get_tf_node_attr(node, a), name=port_name(node.name))
                attr[a] = onnx_tensor
            elif a == "DstT":
                attr["to"] = utils.map_tf_dtype(utils.get_tf_node_attr(node, "DstT"))
            elif a == "SrcT":
                continue
            elif a in ignored_attr:
                continue
            else:
                attr[a] = utils.get_tf_node_attr(node, a)

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
    name = node.input[input_number]

    cast_node = ctx.insert_new_node_on_input(node, "Cast", name)
    cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
    ctx.copy_shape(name, cast_node.output[0])


def no_op(ctx, node, name, args):
    """Skip node."""
    ctx.remove_node(name)


def direct_op(ctx, node, name, args):
    """Take node as is, no updates required"""
    return


def identity_op(ctx, node, name, args):
    """Identity."""
    if node.inputs[0].is_const():
        # should not remove the identity node if it is output of the graph
        if node.output[0] in ctx.outputs:
            return
        # if identity has a const as input, remove it
        input_name = node.input[0]
        output_name = node.output[0]
        ctx.replace_all_inputs(ctx.get_nodes(), output_name, input_name)
        ctx.remove_node(name)


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


def arg_minmax_op(ctx, node, name, args):
    # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
    # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
    axis_node = node.inputs[1]
    axis = axis_node.get_tensor_value()
    if axis < 0:
        # ArgMax|ArgMin in onnx don't necessary support negative axis(not in doc explicitly)
        input_shape = ctx.get_shape(node.input[0])
        dim_count = len(input_shape) if input_shape else 0
        axis = dim_count + axis

    # TF ArgMin/ArgMax may return int32 or int64
    # Onnx ArgMin/ArgMax only supports int64 output, add cast if needed
    if node.get_attr_int("output_type") == onnx_pb.TensorProto.INT32:
        # current node will return int64 after conversion, which differs from previous dtype got from tf
        ctx.set_dtype(node.output[0], onnx_pb.TensorProto.INT64)
        op_name = utils.make_name("Cast")
        cast_node = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name, to=onnx_pb.TensorProto.INT32)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT32)
        ctx.copy_shape(node.output[0], cast_node.output[0])

    node.set_attr("axis", axis)
    node.set_attr("keepdims", 0)
    ctx.remove_input(node, node.input[1])


def reduce_op(ctx, node, name, args):
    axes_node = node.inputs[1]
    axes = axes_node.get_tensor_value()
    if np.isscalar(axes):
        axes = [axes]
    input_shape = ctx.get_shape(node.input[0])
    if input_shape is None:
        if any([val < 0 for val in axes]):
            raise ValueError("reduce_op: cannot have negative axis because we don't know input rank")
    else:
        input_rank = len(ctx.get_shape(node.input[0]))
        axes = [val + input_rank if val < 0 else val for val in axes]

    node.set_attr("axes", axes)
    ctx.remove_input(node, node.input[1])
    keep_dims = node.get_attr("keep_dims")
    if keep_dims:
        del node.attr['keep_dims']
        node.set_attr("keepdims", keep_dims.i)


def square_op(ctx, node, name, args):
    node.type = "Mul"
    node.input.append(node.input[0])


def squeeze_op(ctx, node, name, args):
    # T output = Squeeze(T input, @list(int) squeeze_dims)
    # T squeezed = Squeeze(T data, @AttrType.INTS axes), axes are list of positive integers.
    axis = node.get_attr("axis")
    if not axis:
        axis = node.get_attr("squeeze_dims")
        if axis:
            del node.attr["squeeze_dims"]
    else:
        del node.attr["axis"]

    shape = ctx.get_shape(node.input[0])
    utils.make_sure(shape is not None, "squeeze input shape cannot be None")
    shape_len = len(shape)
    if axis and axis.ints:
        axis = axis.ints
        axis = [a + shape_len if a < 0 else a for a in axis]
    else:
        axis = [i for i, j in enumerate(shape) if j == 1]
    node.set_attr("axes", axis)


def reshape_op(ctx, node, name, args):
    # T output = Reshape(T tensor, Tshape shape, @type Tshape)
    # T reshaped = Reshape(T data, @INTS shape) - but takes a optional 2nd input for shape
    shape_node = node.inputs[1]
    shape = shape_node.get_tensor_value()
    if shape is None:
        log.error("Reshape on node %s does not have a const shape", node.name)
        return
    ctx.remove_input(node, node.input[1])
    node.set_attr("shape", shape)
    ctx.set_shape(node.output[0], shape)


def reshape_op5(ctx, node, name, args):
    dtype = ctx.get_dtype(node.output[0])
    need_casting = dtype in [onnx_pb.TensorProto.INT32,
                             onnx_pb.TensorProto.INT16,
                             onnx_pb.TensorProto.INT64]
    # onnx wants reshape.input[1] to have the value be int64 which is not the case for tensorflow.
    _convert_shapenode_to_int64(ctx, node, 1)
    if ctx.opset >= 8 or not need_casting:
        # onnx reshape can handle the type - done
        return

    # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
    input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
    input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
    ctx.copy_shape(node.output[0], input_cast.output[0])

    # if the next node is already a cast we don't need to insert another one
    next_nodes = ctx.find_output_consumers(node.output[0])
    if len(next_nodes) != 1 or next_nodes[0].type != "Cast":
        op_name = utils.make_name(node.name)
        output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name)
        output_cast.set_attr("to", dtype)
        ctx.set_dtype(output_cast.output[0], dtype)
        ctx.copy_shape(node.output[0], output_cast.output[0])


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

    if node.is_nhwc():
        # transpose input if needed, no need to record shapes on input
        for idx in input_indices:
            parent = node.inputs[idx]
            if node.inputs[idx].is_const():
                # if input is a constant, transpose that one
                if not parent.data_format:
                    val = parent.get_tensor_value(as_list=False)
                    parent.set_tensor_value(val.transpose(NHWC_TO_NCHW))
            else:
                # if input comes from a op, insert transpose op
                input_name = node.input[idx]
                transpose = ctx.insert_new_node_on_input(node, "Transpose", input_name)
                transpose.set_attr("perm", NHWC_TO_NCHW)
                transpose.skip_conversion = True
                shape = ctx.get_shape(input_name)
                new_shape = spatial_map(shape, NHWC_TO_NCHW)
                ctx.set_shape(transpose.output[0], new_shape)
            parent.data_format = "NCHW"

    # kernel must to be transposed
    if with_kernel:
        parent = node.inputs[1]
        need_transpose = True
        if node.inputs[1].is_const():
            # kernel is const - transpose the const if we are the only consumer of const
            # TODO: maybe we should make a copy of the const, or look at the other consumers
            # if they'd want a transose as well.
            consumers = ctx.find_output_consumers(node.input[1])
            if len(consumers) == 1:
                val = parent.get_tensor_value(as_list=False)
                val = val.transpose(HWCN_TO_NCHW)
                parent.set_tensor_value(val)
                parent.data_format = "NCHW"
                need_transpose = False

        if need_transpose:
            input_name = node.input[1]
            transpose = ctx.insert_new_node_on_input(node, "Transpose", input_name)
            transpose.set_attr("perm", HWCN_TO_NCHW)
            transpose.skip_conversion = True
            ctx.copy_shape(input_name, transpose.output[0])
            new_shape = spatial_map(ctx.get_shape(input_name), HWCN_TO_NCHW)
            ctx.set_shape(transpose.output[0], new_shape)
            parent.data_format = "NCHW"

        # some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
        if new_kernel_shape:
            if ctx.opset < 5:
                # old reshape takes new shape as attribute
                input_name = node.input[1]
                reshape = ctx.insert_new_node_on_input(node, "Reshape", input_name)
                reshape.set_attr("shape", new_kernel_shape)
                reshape.skip_conversion = True
            else:
                # new reshape takes new shape as input[1]
                shape_name = utils.make_name(node.name)
                ctx.make_const(shape_name, np.array(new_kernel_shape, dtype=np.int64))
                input_name = node.input[1]
                reshape = ctx.insert_new_node_on_input(node, "Reshape", input_name)
                reshape.input.append(shape_name)
                reshape.skip_conversion = True
            ctx.set_shape(reshape.output[0], new_kernel_shape)

    # transpose outputs if needed
    if node.is_nhwc():
        for idx in output_indices:
            output_name = node.output[idx]
            op_name = utils.make_name(node.name)
            transpose = ctx.insert_new_node_on_output("Transpose", output_name, name=op_name)
            transpose.set_attr("perm", NCHW_TO_NHWC)
            transpose.skip_conversion = True
            ctx.set_shape(transpose.output[0], ctx.get_shape(node.output[idx]))
            node.data_format = "NCHW"


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
            if any(input_shape[i + 2] == -1 for i in range(spatial)):
                log.warning("node %s has unknown dim %s for pads calculation, fallback to auto_pad" % (
                    node.name, str(input_shape)))
                node.set_attr("auto_pad", "SAME_UPPER")
            else:
                for i in range(spatial):
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
    conv_convert_inputs(ctx, node, with_kernel=True)


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

    conv_convert_inputs(ctx, node, with_kernel=True)

    # Note: output_padding, group are left default.


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
    conv_convert_inputs(ctx, node, with_kernel=True, new_kernel_shape=new_kernel_shape)


def pool_op(ctx, node, name, args):
    # T output = MaxPool(T input, @list(int) ksize, @list(int) strides, @string padding, @string data_format)
    # T Y = MaxPool(T X, @AttrType.STRING auto_pad, @AttrType.INTS kernel_shape, @AttrType.INTS pads,
    #               @AttrType.INTS strides)
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

    conv_convert_inputs(ctx, node, with_kernel=False)


def relu6_op(ctx, node, name, args):
    # relu6 = min(max(features, 0), 6)
    node.type = "Relu"
    clip_name = utils.make_name(node.name)
    clip_node = ctx.insert_new_node_on_output("Clip", node.output[0], name=clip_name, min=0.0, max=6.0)
    ctx.copy_shape(node.output[0], clip_node.output[0])
    return [node, clip_node]


def squareddifference_op(ctx, node, name, args):
    node.type = "Sub"
    op_name = utils.make_name(node.name)
    mul = ctx.insert_new_node_on_output("Mul", node.output[0], name=op_name)
    mul.input.append(node.output[0])


def cast_op(ctx, node, name, args):
    # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
    # T2 output = Cast(T1 input, @STRING to)
    dst = node.get_attr("to")
    dst = tf2onnx.utils.ONNX_DTYPE_NAMES[dst]
    node.set_attr("to", dst)


def sign_op4(ctx, node, name, args):
    """Sign op."""
    # T sign = Sign(T Input)
    node_dtype = ctx.get_dtype(node.output[0])
    utils.make_sure(node_dtype, "Dtype of {} is None".format(node.name))
    if node_dtype in [onnx_pb.TensorProto.COMPLEX64, onnx_pb.TensorProto.COMPLEX128]:
        raise ValueError("dtype " + node_dtype + " is not supported in onnx for now")
    input_tensor_type = utils.map_onnx_to_numpy_type(node_dtype)
    zero_name = utils.make_name("{}_zero".format(node.name))
    ctx.make_const(zero_name, np.array(0, dtype=input_tensor_type))
    greater_node = ctx.make_node("Greater", [node.input[0], zero_name])
    less_node = ctx.make_node("Less", [node.input[0], zero_name])
    cast_node_1 = ctx.make_node("Cast", [greater_node.output[0]], {"to": node_dtype})
    cast_node_2 = ctx.make_node("Cast", [less_node.output[0]], {"to": node_dtype})

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    ctx.make_node("Sub", [cast_node_1.output[0], cast_node_2.output[0]], outputs=[node.output[0]],
                  shapes=shapes, dtypes=dtypes)


def sign_op9(ctx, node, name, args):
    node_dtype = ctx.get_dtype(node.output[0])
    utils.make_sure(node_dtype, "Dtype of {} is None".format(node.name))
    if node_dtype in [onnx_pb.TensorProto.BOOL, onnx_pb.TensorProto.COMPLEX64, onnx_pb.TensorProto.COMPLEX128]:
        raise ValueError("dtype " + str(node_dtype) + " is not supported in onnx for now")


def biasadd_op(ctx, node, name, args):
    # T output = BiasAdd(T value, T bias, @string data_format)
    # T output = BiasAddV1(T value, T bias)
    # TODO: for now use add. We may need to convert to NCHW.
    node.type = "Add"
    broadcast_op(ctx, node, name, args)


def biasadd_op7(ctx, node, name, args):
    # T output = BiasAdd(T value, T bias, @string data_format)
    # T output = BiasAddV1(T value, T bias)
    # According TF bias_add definition, the input dim is always only 1.
    node.type = "Add"
    broadcast_op7(ctx, node, name, args)

    # on NHWC, bias will broadcast from largest dim, which is default onnx Add op broadcast behavior.
    if not node.is_nhwc():
        # however, in NCHW, bias should be at 2nd dim, which by default onnx Add op has no way to know,
        # so it needs being reshaped into 3-dim tensor before add
        shape0 = ctx.get_shape(node.input[0])
        shape1 = ctx.get_shape(node.input[1])
        if node.inputs[1].type == 'Const' and len(shape1) == 1:
            new_broadcast_shape = [shape1[0]] + [1] * (len(shape0) - 2)
            shape_name = utils.make_name(node.name)
            ctx.make_const(shape_name, np.array(new_broadcast_shape, dtype=np.int64))
            op_name = node.input[1]
            reshape_node = ctx.insert_new_node_on_input(node, "Reshape", op_name)
            reshape_node.input.append(shape_name)
            ctx.set_shape(reshape_node.output[0], new_broadcast_shape)
            return


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


def _wrap_concat_with_cast(ctx, node):
    """wrap concat in casts for opset < 8 since it only supports."""
    supported_types = [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16]
    dtype = ctx.get_dtype(node.output[0])
    need_casting = dtype not in supported_types
    if need_casting:
        output_name = node.output[0]
        # cast each inputs to float
        for i, inp in enumerate(node.inputs):
            input_cast = ctx.insert_new_node_on_input(node, "Cast", node.input[i])
            input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(input_cast.output[0], onnx_pb.TensorProto.FLOAT)
        next_nodes = ctx.find_output_consumers(node.output[0])
        # cast output back to dtype unless the next op is a cast
        if next_nodes[0].type != "Cast":
            op_name = utils.make_name(node.name)
            output_cast = ctx.insert_new_node_on_output("Cast", output_name, name=op_name)
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.copy_shape(output_name, output_cast.output[0])


def concat_op(ctx, node, name, args):
    # old concat op has axis as input[0]
    axis_node = node.inputs[0]
    axis_val = axis_node.get_tensor_value()
    ctx.remove_input(node, node.input[0])

    if axis_val < 0:  # onnxruntime does not support -1 axis, but TF supports.
        input_shape = ctx.get_shape(node.input[0])
        axis_val = len(input_shape) + axis_val
    node.set_attr("axis", axis_val)

    if ctx.opset < 8:
        # opset < 8: might need to wrap concat in casts since only float is supported
        _wrap_concat_with_cast(ctx, node)
        return


def concatv2_op(ctx, node, name, args):
    # T output = ConcatV2(T values, Tidx axis, @int N, @type Tidx)
    # T concat_result = Concat(T inputs, @INT axis)
    axis_node = node.inputs[-1]
    axis_val = axis_node.get_tensor_value()
    ctx.remove_input(node, node.input[-1])

    if axis_val < 0:  # onnxruntime does not support -1 axis, but TF supports.
        input_shape = ctx.get_shape(node.input[0])
        axis_val = len(input_shape) + axis_val
    node.set_attr("axis", axis_val)

    if ctx.opset < 8:
        # opset < 8: might need to wrap concat in casts since only float is supported
        _wrap_concat_with_cast(ctx, node)
        return


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


def gatherv2_op(ctx, node, name, args):
    # for GatherV2 axis come as input
    axis = node.inputs[2].get_tensor_value()
    ctx.remove_input(node, node.input[2])
    node.set_attr("axis", axis)


def split_op(ctx, node, name, args):
    # T output = Split(int32 split_dim, T value, @int num_split)
    # T outputs = Split(T input, @INT axis, @INTS split)
    split_dims = node.inputs[0].get_tensor_value()
    ctx.remove_input(node, node.input[0])
    node.set_attr("axis", split_dims)


def splitv_op(ctx, node, name, args):
    # T output = SplitV(T value, Tlen size_splits, int32 split_dim, @int num_split, @type Tlen)
    # T outputs = Split(T input, @INT axis, @INTS split)
    split = node.inputs[1].get_tensor_value()
    split_dims = node.inputs[2].get_tensor_value()
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    node.set_attr("split", split)
    node.set_attr("axis", split_dims)


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
        const_val = node.inputs[2].get_tensor_value()
        node.set_attr("value", const_val)
        ctx.remove_input(node, node.input[2])

    ctx.remove_input(node, node.input[1])
    node.set_attr("pads", paddings)

    origin_dtype = ctx.get_dtype(node.output[0])
    if origin_dtype not in [onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.FLOAT,
                            onnx_pb.TensorProto.DOUBLE]:
        cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(name, cast_node.output[0])

        cast_back_node = ctx.insert_new_node_on_output("Cast", node.output[0],
                                                       name=utils.make_name(node.name) + "_castback")
        cast_back_node.set_attr("to", origin_dtype)
        ctx.set_dtype(cast_back_node.output[0], origin_dtype)
        ctx.copy_shape(name, cast_back_node.output[0])


def rsqrt_op(ctx, node, name, args):
    node.type = "Sqrt"
    op_name = utils.make_name(node.name)
    reciprocal = ctx.insert_new_node_on_output("Reciprocal", node.output[0], name=op_name)
    ctx.copy_shape(node.output[0], reciprocal.output[0])


def expanddims_op(ctx, node, name, args):
    # T output = ExpandDims(T input, Tdim dim, @type Tdim)
    # T reshaped = Reshape-1(T data, @ints consumed_inputs, @int64 shape)
    # T expanded = Unsqueeze-1(T data, @ints axes)
    shape = ctx.get_shape(node.output[0])
    if shape is not None and shape.count(-1) < 2:
        # tensorflow already infers the output shape so we can just take it
        shape = ctx.get_shape(node.output[0])
        node.type = "Reshape"
        ctx.remove_input(node, node.input[1])
        node.set_attr("shape", shape)
        return

    # if there is more than one -1 in the shape, Reshape won't support.
    dim_node = node.inputs[1]
    if dim_node.is_const():
        node.type = "Unsqueeze"
        input_rank = len(ctx.get_shape(node.input[0]))
        dim = dim_node.get_tensor_value()
        dim = dim + input_rank + 1 if dim < 0 else dim
        node.set_attr("axes", [dim])
        ctx.remove_input(node, node.input[1])
        return
    raise ValueError("non-const dim is not supported")


def expanddims_op7(ctx, node, name, args):
    # T output = ExpandDims(T input, Tdim dim, @type Tdim), dim is 0-D scalar.
    # T reshaped = Reshape-5(T data, int64 shape)
    # T expanded = Unsqueeze-1(T data, @ints axes)
    shape = ctx.get_shape(node.output[0])
    if shape is not None and shape.count(-1) < 2:
        # tensorflow already infers the output shape so we can just take it
        shape_name = utils.make_name(node.name)
        ctx.make_const(shape_name, np.array(shape, dtype=np.int64))
        node.type = "Reshape"
        node.input[1] = shape_name
        return

    # if there is more than one -1 in the shape, Reshape won't support.
    dim_node = node.inputs[1]
    if dim_node.is_const():
        node.type = "Unsqueeze"
        input_rank = len(ctx.get_shape(node.input[0]))
        dim = dim_node.get_tensor_value()
        dim = dim + input_rank + 1 if dim < 0 else dim
        node.set_attr("axes", [dim])
        ctx.remove_input(node, node.input[1])
        return
    raise ValueError("non-const dim is not supported")


def stridedslice_op(ctx, node, name, args):
    # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
    not_supported_attr = ["ellipsis_mask", "new_axis_mask"]
    for attr_name in not_supported_attr:
        attr = node.get_attr(attr_name)
        if attr is not None and attr.i != 0:
            raise ValueError("StridedSlice: attribute " + attr_name + " not supported")
    input_shape = ctx.get_shape(node.input[0])
    begin = node.inputs[1].get_tensor_value(as_list=False)
    end = node.inputs[2].get_tensor_value(as_list=False)
    strides = node.inputs[3].get_tensor_value(as_list=False)
    max_size = np.iinfo(begin.dtype).max
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask.i if end_mask is not None else 0
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask.i if begin_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
    new_begin = []
    new_end = []
    axes = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    for idx, begin_item in enumerate(begin):
        end_item = end[idx]
        if strides[idx] != 1:
            raise ValueError("StridedSlice: only strides=1 is supported")
        axes.append(idx)

        # an implicit condition is stride == 1 (checked in above)
        if begin_item < 0 and end_item == 0:
            end_item = max_size

        mask = (shrink_axis_mask >> idx) & 1
        if mask != 0:
            new_begin.append(begin_item)
            end_item = begin_item + 1 if begin_item != -1 else max_size
            new_end.append(end_item)
            needs_squeeze.append(idx)
            continue

        mask = (begin_mask >> idx) & 1
        if mask != 0:
            new_begin.append(0)
        else:
            new_begin.append(begin_item)

        mask = (end_mask >> idx) & 1
        if mask != 0:
            new_end.append(max_size)
        else:
            new_end.append(end_item)

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
    if input_dtype != onnx_pb.TensorProto.FLOAT:
        if node.inputs[0].type == "Cast" and len(ctx.find_output_consumers(node.inputs[0].output[0])) == 1:
            # override the previous cast
            cast_node = node.inputs[0]
        else:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
            nodes.insert(0, cast_node)
        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.input[0], cast_node.output[0])
        # undo the cast afer slice
        name = utils.make_name(node.name)
        cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
        cast_node.set_attr("to", input_dtype)
        ctx.set_dtype(cast_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], cast_node.output[0])
        nodes.append(cast_node)


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
        broadcast_op(ctx, mul_op, name, args)


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
    conv_convert_inputs(ctx, node, with_kernel=False)


def upsample_op9(ctx, node, name, args):
    # float32 out = ResizeBilinear/ResizeNearestNeighbor(T images, int size)
    # https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest_neighbor
    # wants the input to be NHWC - adjust target_shape to this.
    ori_shape = ctx.make_node("Shape", [node.input[0]])
    ori_shape_hw = ctx.make_node("Slice", ori_shape.output, {"axes": [0], "starts": [1], "ends": [3]})
    ori_shape_hw_float = ctx.make_node("Cast", ori_shape_hw.output, attr={"to": onnx_pb.TensorProto.FLOAT})

    target_hw = node.inputs[1]
    target_hw_float = ctx.make_node("Cast", target_hw.output, attr={"to": onnx_pb.TensorProto.FLOAT})

    scales_hw = ctx.make_node("Div", [target_hw_float.output[0], ori_shape_hw_float.output[0]])

    const_one_array = ctx.make_const(utils.make_name("one"), np.array([1.0, 1.0]).astype(np.float32))
    # scaler is nchw
    scales = ctx.make_node("Concat", [const_one_array.output[0], scales_hw.output[0]], {"axis": 0})
    input_nchw = ctx.make_node("Transpose", [node.input[0]], {"perm": [0, 3, 1, 2]})
    upsample = ctx.make_node("Upsample", [input_nchw.output[0], scales.output[0]], attr={"mode": args[0]})

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    res = ctx.make_node("Transpose", upsample.output, {"perm": [0, 2, 3, 1]},
                        name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


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
    node.set_attr("sample_size", sample_size)
    ctx.remove_input(node, node.input[1])


def topk_op(ctx, node, name, args):
    # T values, int32 indices = TopKV2(T input, int32 k, @bool sorted=true, @realnumbertype T)
    # T values, I indices = TopK(T x, @int axis=-1, @int k). I: int64
    topk_node_name = node.name
    topk_output1 = node.output[0]
    topk_output2 = node.output[1]

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    k = node.inputs[1].get_tensor_value()
    ctx.remove_node(topk_node_name)
    new_topk_name = utils.make_name(topk_node_name)
    new_topk_node = ctx.make_node("TopK", [node.input[0]],
                                  outputs=[topk_output1, utils.port_name(new_topk_name, 1)],
                                  name=new_topk_name, attr={"k": k},
                                  shapes=shapes, dtypes=[dtypes[0], onnx_pb.TensorProto.INT64])

    new_cast_name = utils.make_name(topk_node_name)
    cast_to_int32 = ctx.make_node("Cast", [new_topk_node.output[1]], outputs=[topk_output2],
                                  name=new_cast_name, attr={"to": onnx_pb.TensorProto.INT32},
                                  shapes=[shapes[1]], dtypes=[onnx_pb.TensorProto.INT32])


def tile_op7(ctx, node, name, args):
    # onnx wants shape input to be int64
    _convert_shapenode_to_int64(ctx, node, 1)


def reorganize_data_op(ctx, node, name, args):
    block_size = node.get_attr("block_size")
    node.set_attr("blocksize", block_size.i)
    conv_convert_inputs(ctx, node, with_kernel=False)


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
        has_correct_shape = has_correct_shape[0]
        for i in needs_broadcast_op:
            input_node = node.inputs[i]
            # get a tensor with zeros (since there is no Fill op as of opset8)
            sub_node = ctx.make_node("Sub", [has_correct_shape, has_correct_shape],
                                     op_name_scope=input_node.name)

            # use add as 'broadcast' op
            add_node = ctx.make_node("Add", [input_node.output[0], sub_node.output[0]],
                                     op_name_scope=input_node.name)
            node.input[i] = add_node.output[0]


def pack_op(ctx, node, name, args):
    # hack to make up for the missing onnx pack op
    axis = node.get_attr("axis").i
    if axis < 0:
        axis += len(ctx.get_shape(node.input[0])) + 1

    inputs = []
    dtype = None
    # insert Unsqueeze on each input
    for i, n in enumerate(node.inputs):
        dtype = ctx.get_dtype(node.input[i])
        shape = ctx.get_shape(node.input[i])
        new_node = ctx.make_node("Unsqueeze", [node.input[i]], op_name_scope=node.name, attr={"axes": [axis]},
                                 shapes=[shape], dtypes=[dtype])
        output_name = new_node.output[0]
        node.input[i] = output_name
        inputs.append(output_name)

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    # concat all unqueezes
    concat = ctx.make_node("Concat", inputs, op_name_scope=node.name, attr={"axis": axis},
                           shapes=shapes, dtypes=dtypes)
    ctx.replace_all_inputs(ctx.get_nodes(), node.output[0], concat.output[0])


def unpack_op(ctx, node, name, args):
    # hack to make up for the missing onnx unpack op
    axis = node.get_attr("axis").i
    # split the tensor into n outputs
    node.type = "Split"
    # for each output we need to squeeze axis
    for n in node.output:
        op_name = utils.make_name(name)
        squeeze_node = ctx.insert_new_node_on_output("Squeeze", n, name=op_name, axes=[axis])
        ctx.copy_shape(n, squeeze_node.output[0])
        ctx.copy_dtype(n, squeeze_node.output[0])


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
    depth = node.inputs[1].get_tensor_value()
    on_val = node.inputs[2].get_tensor_value(as_list=False)
    on = on_val.tolist()
    off = node.inputs[3].get_tensor_value()
    eye = np.eye(depth, dtype=on_val.dtype)
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
        return


def fused_batchnorm_op7(ctx, node, name, args):
    node.type = "BatchNormalization"
    # tf inputs: x, scale, bias, mean, variance
    # tf outputs: y, batch_mean, batch_var
    # a: data_format, epsilon, is_training
    # onnx inputs: X, scale, B, mean, variance, attributes: epsilon, momentum=0.9, spatial : 1
    # output: y, mean, var, savedmean, savedvar,
    # detach unused outputs. While we could let the unused outputs dangle,
    # some runtimes like pytorch/caffe2 do complain about it.
    consumers = [ctx.find_output_consumers(output_name) for output_name in node.output[1:]]
    if not any(consumers):
        new_output = [node.output[0]]
        node.output = new_output

    conv_convert_inputs(ctx, node, with_kernel=False)

    scale_shape = ctx.get_shape(node.input[1])
    mean_shape = ctx.get_shape(node.input[3])
    var_shape = ctx.get_shape(node.input[4])
    val_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[1]))

    if mean_shape != scale_shape:
        new_mean_value = np.array(np.resize(node.inputs[3].get_tensor_value(as_list=False), scale_shape),
                                  dtype=val_type)
        new_mean_node_name = utils.make_name(node.name)
        ctx.make_const(new_mean_node_name, new_mean_value)
        node.input[3] = new_mean_node_name

    if var_shape != scale_shape:
        new_var_value = np.array(np.resize(node.inputs[4].get_tensor_value(as_list=False), scale_shape),
                                 dtype=val_type)
        new_val_node_name = utils.make_name(node.name)
        ctx.make_const(new_val_node_name, new_var_value)
        node.input[4] = new_val_node_name


def matmul_op(ctx, node, name, args):
    # tensorflow allows transpose and conjugated. If found, insert the required transpose.
    # We could use Gemm as well but tensorflow does not pass bias in matmul.
    attrs = ["transpose_a", "transpose_b", "adjoint_a", "adjoint_b", "adj_x", "adj_y"]
    attrs_val = [node.get_attr(attr) for attr in attrs]
    attrs_val = [0 if val is None else val.i for val in attrs_val]

    dtype = ctx.get_dtype(node.output[0])
    if any(attrs_val[2:]):
        # conjugation operation on complex data not supported in onnx for now, so if it's complex than raise exception
        if dtype not in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.DOUBLE]:
            raise ValueError("dtype " + dtype + " is not supported in onnx matmul for now")

    transpose_a = (attrs_val[0] + attrs_val[2] + attrs_val[4]) % 2
    transpose_b = (attrs_val[1] + attrs_val[3] + attrs_val[5]) % 2

    if transpose_a != 0:
        shape = ctx.get_shape(node.input[0])
        if shape:
            perm = list(range(0, len(shape)))
            tmp = perm[-1]
            perm[-1] = perm[-2]
            perm[-2] = tmp
            transpose = ctx.insert_new_node_on_input(node, "Transpose", node.input[0], perm=perm)

    if transpose_b != 0:
        shape = ctx.get_shape(node.input[1])
        if shape:
            perm = list(range(0, len(shape)))
            tmp = perm[-1]
            perm[-1] = perm[-2]
            perm[-2] = tmp
            transpose = ctx.insert_new_node_on_input(node, "Transpose", node.input[1], perm=perm)

    unsupported = ["a_is_sparse", "b_is_sparse"]
    for i in unsupported:
        val = node.get_attr(i)
        if val is not None and val.i != 0:
            raise ValueError(node.type + " attribute " + i + " is not supported")


def fill_op7(ctx, node, name, args):
    # T output = Fill(int32 dims, T value, @int32 index_type)
    # T outputs = Tile(T value, int64 repeats (e.g. dims))
    fill_shape = ctx.get_shape(node.input[0])
    fill_shape_dims = fill_shape[0]
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

    for i in range(fill_shape_dims):
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
    node.input[0] = node.input[1]
    node.input[1] = tmp
    node.type = "Tile"
    ctx.set_dtype(node.output[0], new_dtype)

    if need_cast:
        attr = {"to": val_dtype}
        op_name = utils.make_name(node.name + "/cast_back")
        cast_back = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name, **attr)
        ctx.set_dtype(cast_back.output[0], val_dtype)


def fill_op(ctx, node, name, args):
    node.type = "ConstantOfShape"
    # both shape and value in tensorflow are passed as tensor.
    # In onnx the value is an attribute so we need to fetch the value as const which
    # sooner or later will be a problem for tensorflow-onnx.
    # ConstantOfShape in onnxruntime only support int64, so insert cast op
    input_dtype_is_int64 = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[0])) == np.int64
    if not input_dtype_is_int64:
        cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0], to=onnx_pb.TensorProto.INT64)
    dtype = ctx.get_dtype(node.output[0])
    value = np.array([node.inputs[1].get_tensor_value()]).astype(utils.map_onnx_to_numpy_type(dtype))
    value_proto = numpy_helper.from_array(value)
    node.set_attr("value", value_proto)
    del node.input[1]


def reverse_op8(ctx, node, name, args):
    # T output = ReverseSequence(T input, int32|int64 seq_lengths, @int seq_dim, @int batch_dim)
    # T output = Scan(int64 sequence_lens, variadic initial_state_and_scan_inputs, @graph body,
    #                 @ints directions,@int num_scan_inputs)
    seq_dim = node.get_attr("seq_dim")
    batch_dim = node.get_attr("batch_dim")
    batch_major = seq_dim.i == 1 and (batch_dim or batch_dim.i == 0)
    time_major = batch_dim.i == 1 and (seq_dim or seq_dim.i == 0)
    perm_val = None

    if not batch_major and not time_major:
        error_msg = "unsupported attributes, seq_dim:{}, batch_dim:{}".format(seq_dim, batch_dim)
        raise ValueError(error_msg)

    if time_major:
        old_shape = ctx.get_shape(node.input[0])
        old_dtype = ctx.get_dtype(node.input[0])
        perm_val = [1, 0]
        rank = len(old_shape)
        utils.make_sure(rank >= 2, "rank of reverse_sequence input {} is at least 2".format(node.input[0]))
        perm_val += list(range(2, rank))
        trans_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[0], perm=perm_val)
        new_shape = spatial_map(old_shape, perm_val)
        ctx.set_shape(trans_node.output[0], new_shape)
        ctx.set_dtype(trans_node.output[0], old_dtype)

    # handle batch_major input
    node.type = "Scan"
    node.set_attr("num_scan_inputs", 1)
    input_dtype = ctx.get_dtype(node.input[0])
    input_shape = ctx.get_shape(node.input[0])

    g = ctx.create_new_graph_with_same_config()
    g.make_node('Identity', ['X'], outputs=['Y'])
    g.add_graph_input('X', input_dtype, input_shape[2:])
    g.add_graph_output('Y', input_dtype, input_shape[2:])

    node.set_body_graph_as_attr("body", g)
    node.set_attr("directions", [1])  # reverse the scan input

    seq_len_dtype = ctx.get_dtype(node.input[1])
    if seq_len_dtype != onnx_pb.TensorProto.INT64:
        cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[1])
        cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
        ctx.copy_shape(node.input[1], cast_node.output[0])

    if time_major:
        # get back to time_major
        op_name = utils.make_name(node.name)
        trans_back_node = ctx.insert_new_node_on_output("Transpose", node.output[0],
                                                        name=op_name, perm=perm_val)
        ctx.copy_dtype(node.output[0], trans_back_node.output[0])

    tmp = node.input[0]
    node.input[0] = node.input[1]
    node.input[1] = tmp


def shape_op(ctx, node, name, args):
    # out_type output = Shape(T input, @int32|int64 out_type), out_type by default int32
    # int64 output = Shape(T input)
    dtype = ctx.get_dtype(node.output[0])
    if dtype == onnx_pb.TensorProto.INT64:
        return
    op_name = utils.make_name(node.name)
    output_cast = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name)
    output_cast.set_attr("to", dtype)
    ctx.set_dtype(output_cast.output[0], dtype)
    ctx.copy_shape(node.output[0], output_cast.output[0])


def erf_op(ctx, node, name, args):
    """Error function."""
    # constant names
    a1 = "erf_a1"
    a2 = "erf_a2"
    a3 = "erf_a3"
    a4 = "erf_a4"
    a5 = "erf_a5"
    p = "erf_p"
    one = "erf_one"
    null = "erf_null"

    n = node.name
    output_name = node.output[0]
    erf_a1_node = ctx.get_node_by_output("erf_a1")
    if erf_a1_node is None:
        # insert the constants for erf once
        ctx.make_const(a1, np.array(0.254829592, dtype=np.float32))
        ctx.make_const(a2, np.array(-0.284496736, dtype=np.float32))
        ctx.make_const(a3, np.array(1.421413741, dtype=np.float32))
        ctx.make_const(a4, np.array(-1.453152027, dtype=np.float32))
        ctx.make_const(a5, np.array(1.061405429, dtype=np.float32))
        ctx.make_const(p, np.array(0.3275911, dtype=np.float32))
        ctx.make_const(one, np.array(1., dtype=np.float32))
        ctx.make_const(null, np.array(0., dtype=np.float32))

    x = node.input[0]

    # erf(x):
    #  sign = 1 if x >= 0 else -1
    #  x = abs(x)
    #  # A&S formula 7.1.26
    #  t = 1.0 / (1.0 + p * x)
    #  y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) *  t * math.exp(-x * x)
    #  return sign * y  # erf(-x) = -erf(x)

    x_node = ctx.make_node("Abs", [x], op_name_scope=node.name, name="x")
    negx_node = ctx.make_node("Sub", [null, x], op_name_scope=node.name, name="negx")
    is_positive_node = ctx.make_node("Greater", [x, null], op_name_scope=node.name, name="isPositive")
    is_positive_value_node = ctx.make_node("Cast", is_positive_node.output, op_name_scope=node.name,
                                           name="isPositiveValue", attr={"to": onnx_pb.TensorProto.FLOAT})
    is_neg_node = ctx.make_node("Less", [x, null], op_name_scope=node.name, name="isNeg")
    ig_neg_value_node = ctx.make_node("Cast", is_neg_node.output, op_name_scope=node.name, name="isNegValue",
                                      attr={"to": onnx_pb.TensorProto.FLOAT})
    sign0_node = ctx.make_node("Sub", [is_positive_value_node.output[0], ig_neg_value_node.output[0]],
                               op_name_scope=node.name, name="sign0")
    sign_add_one_node = ctx.make_node("Add", [sign0_node.output[0], one], op_name_scope=node.name, name="signAddOne")
    non_zero_node = ctx.make_node("Abs", sign0_node.output, op_name_scope=node.name, name="nonZero")
    sign_node = ctx.make_node("Sub", [sign_add_one_node.output[0], non_zero_node.output[0]],
                              op_name_scope=node.name, name="sign")
    num_4_node = ctx.make_node("Mul", [x_node.output[0], p], op_name_scope=node.name, name="4")
    num_5_node = ctx.make_node("Add", [num_4_node.output[0], one], op_name_scope=node.name, name="5")
    t_node = ctx.make_node("Div", [one, num_5_node.output[0]], op_name_scope=node.name, name="t")
    xsq_node = ctx.make_node("Mul", [x, negx_node.output[0]], op_name_scope=node.name, name="xsq")
    num_6_node = ctx.make_node("Exp", xsq_node.output, op_name_scope=node.name, name="6")
    num_7_node = ctx.make_node("Mul", [num_6_node.output[0], t_node.output[0]], op_name_scope=node.name, name="7")
    num_8_node = ctx.make_node("Mul", [t_node.output[0], a5], op_name_scope=node.name, name="8")
    num_9_node = ctx.make_node("Add", [num_8_node.output[0], a4], op_name_scope=node.name, name="9")
    num_10_node = ctx.make_node("Mul", [num_9_node.output[0], t_node.output[0]], op_name_scope=node.name, name="10")
    num_11_node = ctx.make_node("Add", [num_10_node.output[0], a3], op_name_scope=node.name, name="11")
    num_12_node = ctx.make_node("Mul", [num_11_node.output[0], t_node.output[0]], op_name_scope=node.name, name="12")
    num_13_node = ctx.make_node("Add", [num_12_node.output[0], a2], op_name_scope=node.name, name="13")
    num_14_node = ctx.make_node("Mul", [num_13_node.output[0], t_node.output[0]], op_name_scope=node.name, name="14")
    num_15_node = ctx.make_node("Add", [num_14_node.output[0], a1], op_name_scope=node.name, name="15")
    num_16_node = ctx.make_node("Mul", [num_15_node.output[0], num_7_node.output[0]], op_name_scope=node.name,
                                name="16")
    num_17_node = ctx.make_node("Sub", [one, num_16_node.output[0]], op_name_scope=node.name, name="17")

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    ctx.make_node("Mul", [num_17_node.output[0], sign_node.output[0]], outputs=[output_name], name=n,
                  shapes=shapes, dtypes=dtypes)


def floordiv_op(ctx, node, name, args):
    # T output = FloorDiv(T x, T y)
    node.type = "Div"
    dtype = ctx.get_dtype(node.input[0])
    if dtype in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.DOUBLE]:
        new_node_name = utils.make_name("floor_div_res")
        floor_res = ctx.insert_new_node_on_output(op_type="Floor", output_name=node.output[0],
                                                  name=new_node_name)
        ctx.copy_dtype(node.output[0], floor_res.output[0])
        ctx.copy_shape(node.output[0], floor_res.output[0])


def floormod_op(ctx, node, name, args):
    # T output = FloorMod(T x, T y)
    div = ctx.make_node(op_type="Div", inputs=node.input)
    dtype = ctx.get_dtype(node.input[0])
    if dtype in [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.DOUBLE]:
        div = ctx.make_node(op_type="Floor", inputs=div.output)

    mul = ctx.make_node(op_type="Mul", inputs=[div.output[0], node.input[1]])
    # res node will take over shape&dtype&output connection info of original "node"
    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    ctx.make_node(op_type="Sub", inputs=[node.input[0], mul.output[0]],
                  name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


def reduce_logic_op(ctx, node, name, args):
    # T output = All(T x, list(int) reduce_indices, @bool keepdims)
    # T output = Any(T x, list(int) reduce_indices, @bool keepdims)
    reduce_dim = node.inputs[1].get_tensor_value()

    # for Any, the reduce_indices can be scalar as observed.
    if np.isscalar(reduce_dim):
        reduce_dim = [reduce_dim]

    utils.make_sure(all(i >= 0 for i in reduce_dim), "negative reduce axis is not supported in onnx for now")

    cast = ctx.make_node(op_type="Cast", inputs=[node.input[0]], attr={"to": onnx_pb.TensorProto.FLOAT})
    keepdims = helper.get_attribute_value(node.get_attr("keep_dims"))
    op_type = "ReduceMin" if node.type == "All" else "ReduceSum"
    reduce_node = ctx.make_node(op_type=op_type, inputs=cast.output, attr={"axes": reduce_dim, "keepdims": keepdims})

    zero_node = ctx.make_const(utils.make_name("zero_reduce"), np.array(0, dtype=np.float32))

    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    ctx.make_node(op_type="Greater", inputs=[reduce_node.output[0], zero_node.output[0]],
                  name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


def zeroslike_op(ctx, node, name, args):
    # T output = ZerosLike(T x)
    # when params "dtype" used, tf will call another op "Fill" instead, so Cast is not needed here.
    input_dtype = ctx.get_dtype(node.input[0])
    node_name = utils.make_name("zero")
    const_zero = ctx.make_const(node_name, np.array(0).astype(utils.map_onnx_to_numpy_type(input_dtype)))
    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(name)
    ctx.make_node(op_type="Mul", inputs=[node.input[0], const_zero.output[0]],
                  name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


def softmax_op(ctx, node, name, args):
    # T output = Softmax(T logits). The axis softmax would be performed on is always on -1.
    # T output = Softmax(T input, @int axis). Default axis is 1.
    logits_rank = len(ctx.get_shape(node.input[0]))
    node.set_attr("axis", logits_rank - 1)


def logical_compare_op(ctx, node, name, args):
    # T2 output = Greater(T1 x, T1 y), T2=tensor(bool)
    # T2 output = Less(T1 x, T1 y), T2=tensor(bool)
    # Great/Less in opset7 only supports limited types, insert Cast if needed
    if ctx.opset < 9:
        supported_dtypes = [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.DOUBLE
        ]
        target_dtype = onnx_pb.TensorProto.FLOAT
        for inp in node.input:
            if ctx.get_dtype(inp) not in supported_dtypes:
                inp_cast = ctx.insert_new_node_on_input(node, "Cast", inp, to=target_dtype)
                ctx.copy_shape(inp, inp_cast.output[0])
                ctx.set_dtype(inp_cast.output[0], target_dtype)


def where_op(ctx, node, name, args):
    # T_y output = Where(T_x condition), return indices of elements whose value are True
    node.type = "NonZero"
    # in onnx, indices are returned in this way [[ind_a_0, ind_b_0, ...], [ind_a_1, ind_b_1,...]];
    # while in tf, the result will be [[ind_a_0, ind_a_1, ...], [ind_b_0, ind_b_1, ...], ...]
    # this is the reason a transpose node inserted here.
    transpose_node = ctx.insert_new_node_on_output("Transpose", node.output[0], name=utils.make_name("where_op_added"))
    ctx.copy_shape(node.output[0], transpose_node.output[0])
    ctx.copy_dtype(node.output[0], transpose_node.output[0])
    return [node, transpose_node]


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
    "Const": (direct_op, []),
    "ConstV2": (direct_op, []),
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
    "GatherNd": (gathernd_op, ["GatherNd"]),
    "Greater": (broadcast_op, []),
    "Identity": (identity_op, ["Identity"]),
    "Less": (broadcast_op, []),
    "Log": (direct_op, []),
    "LogSoftmax": (direct_op, ["LogSoftmax"]),
    "LRN": (lrn_op, []),
    "LogicalAnd": (broadcast_op, ["And"]),
    "LogicalNot": (direct_op, ["Not"]),
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
    "Placeholder": (direct_op, []),
    "PlaceholderV2": (direct_op, []),
    "PlaceholderWithDefault": (direct_op, []),
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
    "Shape": (shape_op, []),
    "Size": (direct_op, []),
    "Sigmoid": (direct_op, []),
    "Slice": (slice_op, []),
    "Split": (split_op, ["Split"]),
    "SplitV": (splitv_op, ["Split"]),
    "Squeeze": (squeeze_op, []),
    "Sqrt": (direct_op, []),
    "Square": (square_op, []),
    "SquaredDifference": (squareddifference_op, []),
    "Softmax": (softmax_op, ["Softmax"]),
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
    "Erf": (erf_op, []),
    "Sign": (sign_op4, []),
    "ZerosLike": (zeroslike_op, []),
}

_OPSET_5 = {
    "ExpandDims": (expanddims_op7, []),
    "OneHot": (onehot_op, []),
    "Reshape": (reshape_op5, []),
    "SparseSoftmaxCrossEntropyWithLogits": (sparse_softmax_cross_entropy_with_logits_op, [])
}

_OPSET_6 = {
    "AddN": (direct_op, ["Sum"]),
    "All": (reduce_logic_op, []),
    "Any": (reduce_logic_op, []),
    "FloorDiv": (floordiv_op, []),
}

_OPSET_7 = {
    "Acos": (direct_op, []),
    "Add": (broadcast_op7, []),
    "Asin": (direct_op, []),
    "Atan": (direct_op, []),
    "BiasAdd": (biasadd_op7, []),
    "BiasAddV1": (biasadd_op7, []),
    "Cast": (direct_op, []),
    "Cos": (direct_op, []),
    "Div": (broadcast_op7, ["Div"]),
    "Equal": (broadcast_op7, []),
    "Fill": (fill_op7, []),
    "FloorMod": (floormod_op, []),
    "FusedBatchNorm": (fused_batchnorm_op7, []),
    "FusedBatchNormV2": (fused_batchnorm_op7, []),
    "Greater": (logical_compare_op, []),
    "Less": (logical_compare_op, []),
    "LogicalAnd": (broadcast_op7, ["And"]),
    "LogicalOr": (broadcast_op7, ["Or"]),
    "MatrixBandPart": (matrixbandpart_op, []),
    "Mul": (broadcast_op7, []),
    "Multinomial": (multinomial_op, []),
    "Pow": (direct_op, []),
    "Range": (range_op7, []),
    "RealDiv": (broadcast_op7, ["Div"]),
    "ResizeBilinear": (upsample_op7, ["Upsample", "linear"]),
    "ResizeNearestNeighbor": (upsample_op7, ["Upsample", "nearest"]),
    "Sin": (direct_op, []),
    "Sub": (broadcast_op7, []),
    "Tan": (direct_op, []),
    "Tile": (tile_op7, []),
    "TruncateDiv": (broadcast_op7, ["Div"]),

    # workaround created ONNX node in pre-rewriters to make sure those ops
    # is handled especially on their contained subgraphs.
    "If": (direct_op, []),
    "Loop": (direct_op, []),
    "Scan": (direct_op, []),
}

_OPSET_8 = {
    "ReverseSequence": (reverse_op8, []),  # make use of scan
    "Select": (select_op8, []),
}

_OPSET_9 = {
    "Acosh": (direct_op, []),
    "Asinh": (direct_op, []),
    "Atanh": (direct_op, []),
    "Cosh": (direct_op, []),
    "Erf": (direct_op, []),
    "Fill": (fill_op, []),
    "Greater": (logical_compare_op, []),
    "IsNan": (direct_op, ["IsNaN"]),
    "Less": (logical_compare_op, []),
    "ResizeBilinear": (upsample_op9, ["Upsample", "linear"]),
    "ResizeNearestNeighbor": (upsample_op9, ["Upsample", "nearest"]),
    "Sign": (sign_op9, []),
    "Sinh": (direct_op, []),
    "Where": (where_op, []),
}

_OPSETS = [
    (4, _OPSET_4),
    (5, _OPSET_5),
    (6, _OPSET_6),
    (7, _OPSET_7),
    (8, _OPSET_8),
    (9, _OPSET_9),
]


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
        for n in set(match.get_nodes()):
            if n != output:
                g.remove_node(n.name)
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
        mean = output.inputs[1].get_tensor_value()
        dtype = g.get_dtype(output.output[0])
        op_name = utils.make_name("RandomNormal")
        out_name = port_name(op_name)

        rn_op = match.get_op('input1')
        if rn_op.inputs[0].type == "Shape":
            shape_node = rn_op.inputs[0]
            new_node = g.make_node("RandomNormalLike", [shape_node.input[0]], outputs=[out_name], name=op_name,
                                   attr={"mean": mean, "scale": 1.0, "dtype": dtype})
        else:
            shape = g.get_shape(output.output[0])
            new_node = g.make_node("RandomNormal", [], outputs=[out_name], name=op_name,
                                   attr={"shape": shape, "mean": mean, "scale": 1.0, "dtype": dtype})

        g.replace_all_inputs(ops, output.output[0], new_node.output[0])
        for n in set(match.get_nodes()):
            g.remove_node(n.name)
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
        new_node = g.make_node(
            "Dropout",
            [inputs2.input[0]],
            outputs=[out_name],
            name=op_name,
            attr={"ratio": 1.0},
            shapes=[g.get_shape(inputs2.input[0])],
            dtypes=[g.get_dtype(inputs2.input[0])]
        )
        g.replace_all_inputs(ops, outputs.output[0], new_node.output[0])
        for n in set(match.get_nodes()):
            g.remove_node(n.name)

    return ops


def rewrite_flatten(g, ops):
    pattern_fixed_shape_input = \
        OpTypePattern('Reshape', name='reshape', inputs=[
            OpTypePattern("*", name="input"),
            OpTypePattern('Pack', name="pack", inputs=[
                OpTypePattern('StridedSlice', name="slice", inputs=[
                    "*", "*", "*", "*",
                ]),
                "*",
            ]),
        ])
    pattern_non_fixed_shape_input = \
        OpTypePattern('Reshape', name='reshape', inputs=[
            OpTypePattern("*", name="input"),
            OpTypePattern('Pack', name="pack", inputs=[
                OpTypePattern('StridedSlice', name="slice", inputs=[
                    OpTypePattern('Shape', inputs=[
                        OpTypePattern("*", name="input2")
                    ]),
                    "*", "*", "*",
                ]),
                "*",
            ]),
        ])
    matcher = GraphMatcher(pattern_fixed_shape_input)
    match_results_1 = list(matcher.match_ops(ops))

    matcher = GraphMatcher(pattern_non_fixed_shape_input)
    match_results_2 = list(matcher.match_ops(ops))

    match_results = [(match_results_1, True), (match_results_2, False)]
    for match_results, check_fixed_input_shape in match_results:
        for match in match_results:
            input_node = match.get_op('input')
            reshape_node = match.get_op('reshape')
            pack_node = match.get_op('pack')
            slice_node = match.get_op('slice')
            need_rewrite = pack_node.inputs[1].is_const() and pack_node.inputs[1].get_tensor_value() == -1
            if not need_rewrite:
                continue

            input_shape = g.get_shape(reshape_node.input[0])
            need_rewrite = input_shape is not None
            if not need_rewrite:
                continue

            if check_fixed_input_shape:
                need_rewrite = slice_node.inputs[0].is_const() and \
                               np.array_equal(list(input_shape), list(slice_node.inputs[0].get_tensor_value()))
                if not need_rewrite:
                    continue

            begin = slice_node.inputs[1].get_tensor_value(as_list=False)
            end = slice_node.inputs[2].get_tensor_value(as_list=False)
            strides = slice_node.inputs[3].get_tensor_value(as_list=False)
            need_rewrite = np.array_equal(begin, [0]) and len(end) == 1 and \
                           np.array_equal(strides, [1]) and end[0] - begin[0] == len(input_shape) - 2
            if not need_rewrite:
                continue

            op_name = utils.make_name("Flatten")
            out_name = port_name(op_name)
            new_node = g.make_node("Flatten", [reshape_node.input[0]], outputs=[out_name], name=op_name)

            last_dim = input_shape[-1]
            sec_last_dim = input_shape[-2]
            new_dim = None
            if last_dim > 0 and sec_last_dim > 0:
                new_dim = last_dim * sec_last_dim
            else:
                new_dim = -1

            g.set_shape(out_name, input_shape[:-2] + [new_dim])
            g.replace_all_inputs(ops, reshape_node.output[0], out_name)

            for n in set(match.get_nodes()):
                if n != input_node:
                    g.remove_node(n.name)

    return ops


def rewrite_constant_fold(g, ops):
    """
    We call tensorflow transform with constant folding but in some cases tensorflow does
    fold all constants. Since there are a bunch of ops in onnx that use attributes where
    tensorflow has dynamic inputs, we badly want constant folding to work. For cases where
    tensorflow missed something, make another pass over the graph and fix want we care about.
    """
    func_map = {
        "Add": np.add,
        "GreaterEqual": np.greater_equal,
        "Cast": np.cast,
        "ConcatV2": np.concatenate,
        "Less": np.less,
        "ListDiff": np.setdiff1d,
        "Mul": np.multiply,
        "Pack": np.stack,
        "Range": np.arange,
        "Sqrt": np.sqrt,
        "Sub": np.subtract,
    }
    ref_cnt_per_node = {}
    for idx, op in enumerate(ops):
        for op_input in op.inputs:
            if op_input.name not in ref_cnt_per_node:
                ref_cnt_per_node[op_input.name] = 0
            ref_cnt_per_node[op_input.name] += 1

    # pylint: disable=too-many-nested-blocks
    keep_looking = True
    while keep_looking:
        keep_looking = False
        for idx, op in enumerate(ops):
            func = func_map.get(op.type)
            if func is None:
                continue
            try:
                inputs = []
                for node in op.inputs:
                    if not node.is_const():
                        break
                    inputs.append(node.get_tensor_value(as_list=False))

                log.debug("op name %s, %s, %s", op.name, len(op.input), len(inputs))
                if inputs and len(op.input) == len(inputs):
                    log.info("folding node type=%s, name=%s" % (op.type, op.name))
                    if op.type == "Cast":
                        dst = op.get_attr_int("to")
                        np_type = tf2onnx.utils.map_onnx_to_numpy_type(dst)
                        val = np.cast[np_type](*inputs)
                    elif op.type == "ConcatV2":
                        axis = inputs[-1]
                        values = inputs[:-1]
                        val = func(tuple(values), axis)
                    elif op.type == "ListDiff":
                        out_type = op.get_attr_int("out_idx")
                        np_type = tf2onnx.utils.map_onnx_to_numpy_type(out_type)
                        val = func(*inputs)
                        val = val.astype(np_type)
                    elif op.type in ["Pack"]:
                        # handle ops that need input array and axis
                        axis = op.get_attr_int("axis")
                        val = func(inputs, axis=axis)
                    elif op.type == "Range":
                        dtype = op.get_attr_int("Tidx")
                        np_type = tf2onnx.utils.map_onnx_to_numpy_type(dtype)
                        val = func(*inputs, dtype=np_type)
                    else:
                        val = func(*inputs)

                    new_node_name = utils.make_name(op.name)
                    new_output_name = new_node_name
                    old_output_name = op.output[0]
                    old_node_name = op.name
                    log.debug("create const node [%s] replacing [%s]", new_node_name, old_node_name)
                    ops[idx] = g.make_const(new_node_name, val)
                    ref_cnt_per_node[new_node_name] = ref_cnt_per_node[old_node_name]

                    log.debug("replace old output [%s] with new output [%s]", old_output_name, new_output_name)
                    # need to re-write the consumers input name to use the const name
                    consumers = g.find_output_consumers(old_output_name)
                    if consumers:
                        for consumer in consumers:
                            g.replace_input(consumer, old_output_name, new_output_name)
                    for node in op.inputs:
                        ref_cnt_per_node[node.name] -= 1
                        if ref_cnt_per_node[node.name] == 0:
                            g.remove_node(node.name)
                    # keep looking until there is nothing we can fold.
                    # We keep the graph in topological order so if we folded,
                    # the result might help a following op.
                    keep_looking = True
            except Exception as ex:
                tb = traceback.format_exc()  # pylint: disable=bare-except
                log.info("exception: %s, details: %s", ex, tb)
                # ignore errors

        # pylint: enable=too-many-nested-blocks
    return ops


def rewrite_logical_compare_with_equal(g, ops):
    patterns = {"GreaterEqual": "Greater",
                "LessEqual": "Less"}
    for p in patterns:
        pattern = OpTypePattern(p, name='compare_with_equal')
        compare_name = patterns[p]
        matcher = GraphMatcher(pattern)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            nodes_to_append = []
            compare_e_op = match.get_op('compare_with_equal')
            data_type = g.get_dtype(compare_e_op.input[0])
            compare_input_ids = compare_e_op.input
            need_cast = g.opset < 9 and data_type not in (onnx_pb.TensorProto.FLOAT16,
                                                          onnx_pb.TensorProto.FLOAT,
                                                          onnx_pb.TensorProto.DOUBLE)
            if need_cast:
                compare_input_ids = []
                for input_id in compare_e_op.input:
                    cast_node = g.make_node("Cast", [input_id], op_name_scope=compare_e_op.name,
                                            attr={"to": onnx_pb.TensorProto.FLOAT}, shapes=[g.get_shape(input_id)],
                                            dtypes=[onnx_pb.TensorProto.FLOAT])
                    compare_input_ids.append(cast_node.output[0])
                    nodes_to_append.append(cast_node)

            g_node = g.make_node(compare_name, compare_input_ids, op_name_scope=compare_e_op.name,
                                 dtypes=[onnx_pb.TensorProto.BOOL])
            set_shape_from_inputs_broadcast(g, compare_input_ids, g_node.output[0])
            new_shape = g.get_shape(g_node.output[0])
            nodes_to_append.append(g_node)

            e_node = g.make_node("Equal", compare_e_op.input, op_name_scope=compare_e_op.name,
                                 shapes=[new_shape], dtypes=[onnx_pb.TensorProto.BOOL])
            nodes_to_append.append(e_node)

            compare_e_op.type = "LogicalOr"
            compare_e_op.input[0] = g_node.output[0]
            compare_e_op.input[1] = e_node.output[0]
            g.set_dtype(compare_e_op.output[0], onnx_pb.TensorProto.BOOL)
            g.set_shape(compare_e_op.output[0], new_shape)
            ops.extend(nodes_to_append)
    return ops


def rewrite_incomplete_type_support(g, ops, impacted_ops):
    """
    for ops that have inclomplete type support, insert casts.
    This is needed for some tensor ops in opset7 and for some ops in winml-rs5.
    It is not helping performance but better than the model not working at all.
    """
    ignored_input_index = {
        "Tile": [1],  # Tile's second input can only be int64
    }
    new_ops = []
    org_ops = [n for n in ops]
    for op in org_ops:
        if op.type in impacted_ops:
            cast_inserted = []
            output_dtype = None
            ignored_inputs = ignored_input_index.get(op.type)
            # insert casts on inputs if the runtime only supports float
            for i, input_node in enumerate(op.inputs):
                if ignored_inputs and i in ignored_inputs:
                    continue

                input_name = op.input[i]
                dtype = g.get_dtype(input_name)
                if dtype is None:
                    log.warning("adding Cast for op %s (type is %s)' input: %s, dtype should not be None",
                                op.name, op.type, input_name)

                if dtype != onnx_pb.TensorProto.FLOAT:
                    output_dtype = dtype
                    log.debug("insert cast for node %s on input %s", op.name, input_name)
                    if input_node and input_node.type == "Cast" \
                            and len(g.find_output_consumers(input_node.output[0])) == 1:
                        input_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
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
                    log.debug("insert cast back for node %s on output %s [dtype=%s]", op.name, output_name,
                              output_dtype)
                    output_cast = g.insert_new_node_on_output("Cast", output_name, name=name)
                    output_cast.set_attr("to", output_dtype)
                    g.set_dtype(output_cast.output[0], output_dtype)
                    g.copy_shape(output_name, output_cast.output[0])
                    cast_inserted.append(output_cast)

            if cast_inserted:
                new_ops.extend(cast_inserted)
        new_ops.append(op)
    return new_ops


def rewrite_incomplete_type_support_rs5(g, ops):
    return rewrite_incomplete_type_support(g, ops, ["Unsqueeze", "Mul", "Concat", "Slice", "Transpose"])


def rewrite_incomplete_type_support_rs6(g, ops):
    return rewrite_incomplete_type_support(g, ops, ["Div", "IsNaN", "ReduceSum", "Slice", "Split", "Tile", "Transpose"])


def rewrite_conv2d_with_pad(g, ops):
    pattern = \
        OpTypePattern("Conv2D", name="conv", inputs=[
            OpTypePattern("Pad", name="pad"),
            OpTypePattern("*")
        ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        conv = match.get_op("conv")
        pad = match.get_op("pad")
        paddings = pad.inputs[1]

        if not paddings.is_const():
            continue
        mode = pad.get_attr("mode")
        if mode:
            mode = mode.s.decode("utf-8").lower()
        if mode not in [None, "constant"] or len(pad.input) >= 3:
            continue
        # Conv2D already has a pad
        if conv.get_attr("padding") == "SAME":
            continue

        log.debug("merge pad [%s] into conv [%s]", pad.name, conv.name)
        paddings_val = np.array(paddings.get_tensor_value())
        # can't pad on batch or channel dimensions
        if np.any(paddings_val[0]) or np.any(paddings_val[3]):
            continue

        paddings_val = paddings_val[1:3]
        paddings_val = paddings_val.transpose().flatten()
        g.replace_input(conv, conv.input[0], pad.input[0])
        # convert Conv2D
        conv.type = "Conv"
        conv_op(g, conv, conv.name, [])
        conv.skip_conversion = True
        conv.set_attr("auto_pad", "NOTSET")
        conv.set_attr("pads", paddings_val)
    return ops


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
        custom_opset = {k: v for k, v in custom_op_handlers.items()}
        ops_mapping.update(custom_opset)

    ops = [n for n in g.get_nodes()]
    for node in ops:
        if node.need_skip():
            log.debug("explictly skip node " + node.name)
            continue

        op = node.type
        map_info = ops_mapping.get(op)
        if map_info is None:
            if continue_on_error:
                unmapped_op[op] += 1
                continue
            else:
                raise ValueError("tensorflow op " + op + " is not supported")
        mapped_op[op] += 1
        func, args = map_info
        if args:
            node.type = args[0]
            args = args[1:]
        try:
            body_graphs = node.get_body_graphs()
            if body_graphs:
                for attr, b_g in body_graphs.items():
                    log.debug("start handling subgraph of %s's attribute %s", node.name, attr)
                    b_g.topological_sort(b_g.get_nodes())
                    # we assume only ONNX nodes have subgraph defined in pre-rewriters.
                    # that means, if we create node having subgraphs in this step, the
                    # created subgraphs' nodes won't be mapped.
                    m_ops, unm_ops = tensorflow_onnx_mapping(b_g, continue_on_error, custom_op_handlers)
                    mapped_op += m_ops
                    unmapped_op += unm_ops
                    log.debug("finish handling subgraph of %s's attribute %s", node.name, attr)

            func(g, node, node.name, args)
            node.skip_conversion = True
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            log.error("node %s: exception %s" % (node.name, ex))
            ex_ext = traceback.format_exception(type_, value_, traceback_)
            if continue_on_error:
                log.info(ex_ext)
            else:
                raise ex

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
                ctx.copy_shape(output_name, transpose.output[0])
                ctx.set_shape(output_name, np.array(shape)[NHWC_TO_NCHW])
                ops.append(transpose)
                ops.append(node)
                continue
            ops.append(node)
    ctx.reset_nodes(ops)


def tf_optimize(inputs, outputs, graph_def, fold_constant=None):
    """Optimize tensorflow graph for inference."""
    transforms = []
    if fold_constant:
        transforms.extend([
            "fold_constants(ignore_errors=true)",
            "remove_attribute(attribute_name=_class)",  # remove node colocation attributes
        ])

    transforms.extend([
        "fold_batch_norms",
        "fold_old_batch_norms",
    ])
    needed_names = [utils.node_name(i) for i in inputs] + [utils.node_name(i) for i in outputs]
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)
    graph_def = TransformGraph(graph_def, inputs, outputs, transforms)
    return graph_def


def topological_sort(g, continue_on_error):
    ops = g.get_nodes()
    if not continue_on_error:
        g.topological_sort(ops)
    else:
        try:
            g.topological_sort(ops)
        except:  # pylint: disable=bare-except
            # if we continue on error, ignore graph cycles so we can report all missing ops
            pass


def run_rewriters(g, funcs, continue_on_error):
    """Rewrite the original graph and body graphs of nodes"""
    # NOTE(wayuanho):
    # 1. we don't sort graph here, rewriter is expected to do it on its own.
    # 2. the graph here may have circles, current topological_sort cannot handle it.
    for func in funcs:
        try:
            ops = func(g, g.get_nodes())
            g.reset_nodes(ops)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            log.error("rewriter %s: exception %s", func, ex)
            ex_ext = traceback.format_exception(type_, value_, traceback_)
            if continue_on_error:
                log.info(ex_ext)
            else:
                raise ex

    if g.contained_graphs:
        for dict_val in g.contained_graphs.values():
            for attr_name, b_g in dict_val.items():
                run_rewriters(b_g, funcs, attr_name)


def process_tf_graph(tf_graph, continue_on_error=False, verbose=False, target=None,
                     opset=None, custom_op_handlers=None, custom_rewriter=None,
                     extra_opset=None, shape_override=None, inputs_as_nchw=None,
                     input_names=None, output_names=None):
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
            input_names: list of input node names in graph, input name format as node_name:port_id
            output_names: list of output node names in graph, output name format as node_name:port_id
        Return:
            onnx graph
    """
    if shape_override is None:
        shape_override = {}
    if inputs_as_nchw is None:
        inputs_as_nchw = []
    if target is None:
        target = DEFAULT_TARGET

    onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes = tensorflow_to_onnx(tf_graph, shape_override)

    io_to_check = []
    if input_names:
        io_to_check.extend(input_names)
    if output_names:
        io_to_check.extend(output_names)

    if io_to_check:
        # check output existence in case user passed in wrong output ids
        non_exists = set(io_to_check) - set(output_shapes.keys())
        if non_exists:
            log.error("\nFailed to convert: inputs/outputs specified do not exist, make sure your passed"
                      "in format: input/output_node_name:port_id. Problematical inputs/outputs are: %s \n",
                      non_exists)
            raise ValueError("Inputs/Outputs Not Found")

    g = Graph(onnx_nodes, output_shapes, dtypes, target, opset, extra_opset, output_names)

    infer_shape_for_graph(g)

    if inputs_as_nchw:
        transpose_inputs(g, inputs_as_nchw)

    # pre-processing graph rewrites
    # bi-directional re-writer should be placed after single directional re-writer
    rewriters = [rewrite_transpose, rewrite_flatten,
                 rewrite_random_uniform, rewrite_random_uniform_fold_const,
                 rewrite_random_normal, rewrite_dropout,
                 rewrite_leakyrelu, rewrite_conv2d_with_pad,
                 rewrite_single_direction_lstm, rewrite_bi_direction_lstm,
                 rewrite_single_direction_gru, rewrite_single_direction_grublock,
                 rewrite_bi_direction_gru, rewrite_logical_compare_with_equal,
                 rewrite_custom_rnn_cell, rewrite_generic_loop, rewrite_cond
                 ]

    if custom_rewriter is not None:
        rewriters.extend(custom_rewriter)

    run_rewriters(g, rewriters, continue_on_error)

    # some nodes may already copied into inner Graph, so remove them from main Graph.
    g.delete_unused_nodes(output_names)
    topological_sort(g, continue_on_error)

    if custom_op_handlers is None:
        custom_op_handlers = {}
    mapped_op, unmapped_op = tensorflow_onnx_mapping(g, continue_on_error, custom_op_handlers)

    # post-processing rewriters
    late_rewriters = []
    if TARGET_RS5 in target:
        late_rewriters.append(rewrite_incomplete_type_support_rs5)
    if TARGET_RS6 in target:
        late_rewriters.append(rewrite_incomplete_type_support_rs6)
    if late_rewriters:
        run_rewriters(g, late_rewriters, continue_on_error)

    # onnx requires topological sorting
    topological_sort(g, continue_on_error)

    g.update_proto()

    if verbose:
        print("tensorflow ops: {}".format(op_cnt))
        print("tensorflow attr: {}".format(attr_cnt))
        print("onnx mapped: {}".format(mapped_op))
        print("onnx unmapped: {}".format(unmapped_op))

    return g
