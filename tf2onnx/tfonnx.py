# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - rewrite tensorflow graph to onnx graph
"""
import logging
import collections

from onnx import ModelProto, helper
import tf2onnx
from tf2onnx import utils
from tf2onnx.graph_matcher import *
from tf2onnx.graph import Node, Graph
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools.freeze_graph import freeze_graph

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx")


FLAVOR = ["onnxmsrt"]


def tensorflow_to_onnx(graph):
    """
    Load tensorflow graph into an onnx graph with minimal rewrites so
    we can use the onnx graph as intermediate graph.
    """

    # ignore the following attributes
    ignored_attr = ["unknown_rank", "_class", "Tidx", "Tshape", "use_cudnn_on_gpu", "Index",
                    "Tpaddings", "TI", "Tparams", "Tindices", "Tlen", "Tdim", "dynamic_size", "element_shape",
                    "Tmultiples", "output_dtype", "Tblock_shape", "Tcrops"]
    # some stats
    op_cnt = collections.Counter()
    attr_cnt = collections.Counter()
    onnx_nodes = []
    output_shapes = {}
    dtypes = {}

    # find outputs
    ops = graph.get_operations()

    # create dict with output to shape mappings
    for node in ops:
        for out in node.outputs:
            try:
                shape = out.get_shape().as_list()
            except Exception as ex:
                shape = []
            output_shapes[out.name] = shape

    # minimal conversion of attributes
    for node in ops:
        attr = {}
        takeit = True
        op_cnt[node.type] += 1
        for a in node.node_def.attr:
            attr_cnt[a] += 1
            if a == "dtype":
                attr[a] = utils.get_tf_dtype(node)
            elif a == "T":
                dtype = node.get_attr("T")
                if dtype:
                    if not isinstance(dtype, list):
                        dtypes[node.name] = utils.TF_TO_ONNX_DTYPE.get(dtype)
            elif a == "output_type":
                out_type = node.get_attr("output_type")
                out_type = utils.TF_TO_ONNX_DTYPE[out_type]
                attr[a] = out_type
            elif a == "out_type":
                out_type = node.get_attr("out_type")
                out_type = utils.TF_TO_ONNX_DTYPE[out_type]
                attr[a] = out_type
            elif a == "shape":
                attr[a] = utils.get_shape(node)
            elif a == "Tperm":
                pass
            elif a == "_output_shapes":
                attr[a] = utils.get_shape(node)
            elif a == "value":
                onnx_tensor = utils.tf_to_onnx_tensor(node.get_attr(a), name=node.name + ":0")
                attr[a] = onnx_tensor
            elif a == "DstT":
                dst = node.get_attr("DstT")
                dst = tf2onnx.utils.TF_TO_ONNX_DTYPE[dst]
                dst = tf2onnx.utils.ONNX_DTYPE_NAMES[dst]
                attr["to"] = dst
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
                raise ex

    return onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes

# pylint: disable=W0613,C0111,W0612


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
    return node


def broadcast_op(ctx, node, name, args):
    """Elementwise Ops with broadcast flag."""
    shape0 = ctx.get_shape(node.input[0])
    shape1 = ctx.get_shape(node.input[1])
    if shape0 != shape1:
        node.set_attr("broadcast", 1)
        if "onnxmsrt" in FLAVOR or "caffe2" in FLAVOR:
            if shape0 is not None and len(shape0) == 0:
                if node.inputs[0].is_const():
                    shape0 = node.inputs[0].scalar_to_dim1()
            if shape1 is not None and len(shape1) == 0:
                if node.inputs[1].is_const():
                    shape1 = node.inputs[1].scalar_to_dim1()
            # caffe2 and onnxmsrt broadcast left to right only - swap if possible
            if shape0 and shape1 and len(shape0) < len(shape1) and node.type in ["Mul", "Add"]:
                tmp = node.input[0]
                node.input[0] = node.input[1]
                node.input[1] = tmp
    else:
        node.set_attr("broadcast", 0)
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
    input_node = helper.make_tensor_value_info(node.output[0], node.dtype, node.shape)
    ctx.model_inputs.append(input_node)
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


def shape_op(ctx, node, name, args):
    # FIXME - this is not correct
    shape = ctx.get_shape(node.input[0])
    if shape[0] is None or shape[0] == -1:
        shape[0] = 1
    old_output = node.output[0]
    node_name = utils.make_name(node.name)
    new_node = ctx.make_const(node_name, "Const", np.zeros(shape, dtype=np.float32))
    new_node.output.append(node_name + ":0")
    for n in ctx.get_nodes():
        for i, input_name in enumerate(n.input):
            if input_name == old_output:
                n.input[i] = node_name + ":0"
                break
    return new_node


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]


def conv_convert_inputs(ctx, node, with_kernel=False, new_kernel_shape=None):
    """Convert input and kernel from tensorflow to onnx. This maybe require to
        to insert transpose ops for input, kernel and output unless they are constants
        and we can transpose the constant.
        We transpose inputs if they are in NHWC. We always transpsoe the kernel from
        HWNC to NCHW. Outputs are transposed if the format is NHWC.
        Some convolutions like depthwise_conv2d require a reshape of the kernel.
        Args:
            ctx: the parent graph
            node: node of the convolution op
            with_kernel: transpose the kernel
            new_kernel_shape: reshape the kernel
    """
    def calc_shape(a, b):
        if a and b:
            return [a[b[i]] for i in b]
        return None

    nodes = []

    if node.is_nhwc():
        # transpose input if needed, no need to record shapes on input
        if node.inputs[0].is_const():
            # if input is a constant, transpose that one
            parent = node.inputs[0]
            if not parent.data_format:
                val = parent.get_tensor_value()
                parent.set_tensor_value(val.transpose(NHWC_TO_NCHW))
                parent.data_format = "NCHW"
        else:
            # if input comes from a op, insert transpose op
            input_name = node.input[0]
            op_name = utils.make_name(node.name)
            transpose = ctx.insert_new_node_on_input(node, "Transpose", input_name, name=op_name)
            transpose.set_attr("perm", NHWC_TO_NCHW)
            transpose.inserted_nchw = True
            ctx.set_shape(transpose.output[0], calc_shape(ctx.get_shape(input_name), NHWC_TO_NCHW))
            nodes.append(transpose)

    # kernel mist to be transposed
    if with_kernel:
        if node.inputs[1].is_const():
            # kernel is const - transpose the const
            parent = node.inputs[1]
            if not parent.data_format:
                val = parent.get_tensor_value()
                val = val.transpose(HWCN_TO_NCHW)
                parent.set_tensor_value(val)
                parent.data_format = "NCHW"
        else:
            # kernel comes from op, insert transpose op
            op_name = utils.make_name(node.name)
            input_name = node.input[1]
            transpose = ctx.insert_new_node_on_input(node, "Transpose", input_name, name=op_name)
            transpose.set_attr("perm", HWCN_TO_NCHW)
            transpose.inserted_nchw = True
            ctx.copy_shape(input_name, transpose.output[0])
            ctx.set_shape(transpose.output[0], calc_shape(ctx.get_shape(input_name), HWCN_TO_NCHW))
            nodes.append(transpose)

        # some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
        if new_kernel_shape:
            op_name = utils.make_name(node.name)
            input_name = node.input[1]
            reshape = ctx.insert_new_node_on_input(node, "Reshape", input_name, name=op_name)
            reshape.set_attr("shape", new_kernel_shape)
            ctx.set_shape(reshape.output[0], new_kernel_shape)
            nodes.append(reshape)

    # insert conv node after inputs
    nodes.append(node)

    # transpose outputs if needed
    if node.is_nhwc():
        # TODO: what if len(output) > 0 ?
        for i, output_name in enumerate(node.output):
            op_name = utils.make_name(node.name)
            transpose = ctx.insert_new_node_on_output("Transpose", output_name, name=op_name)
            transpose.set_attr("perm", NCHW_TO_NHWC)
            transpose.inserted_nchw = True
            ctx.set_shape(transpose.output[0], calc_shape(ctx.get_shape(node.output[0]), NCHW_TO_NHWC))
            nodes.append(transpose)
    return nodes


def add_padding(node, kernel_shape, strides):
    padding = node.get_attr("padding")
    if padding:
        padding = padding.s.decode("utf-8")
        if padding == 'SAME':
            s_h, s_w = strides[0], strides[1]
            k_h, k_w = kernel_shape[0], kernel_shape[1]
            p_x0 = (k_w - s_w) // 2
            p_y0 = (k_h - s_h) // 2
            p_x1 = k_w - s_w - p_x0
            p_y1 = k_h - s_h - p_y0
            node.set_attr("pads", [p_y0, p_x0, p_y1, p_x1])
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


def conv_kernel_shape(ctx, node, input_idx):
    kernel_shape = ctx.get_shape(node.input[1])
    if len(kernel_shape) != 4:
        raise ValueError("only Conv2D is supported")
    h, w, c, n = kernel_shape
    kernel_shape = [h, w]
    node.set_attr("kernel_shape", kernel_shape)
    return kernel_shape


def conv_op(ctx, node, name, args):
    # T output = Conv2D(T input, T filter, @list(int) strides, @bool use_cudnn_on_gpu,
    #                       @string padding, @string data_format)
    # T Y = Conv(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
    #                       @AttrType.INTS kernel_shape, @AttrType.INTS pads, @AttrType.INTS strides)
    kernel_shape = conv_kernel_shape(ctx, node, 1)
    strides = conv_dims_attr(node, "strides")
    conv_dims_attr(node, "dilations")
    add_padding(node, kernel_shape, strides)

    nodes = conv_convert_inputs(ctx, node, with_kernel=True)
    return nodes


def convtranspose_op(ctx, node, name, args):
    # T output = Conv2DBackpropInput(int32 input_sizes, T filter, T out_backprop,
    #    @list(int) strides, @bool use_cudnn_on_gpu, @string padding, @string data_format, @list(int) dilations)
    # T Y = ConvTranspose(T X, T W, T B, @STRING auto_pad, @INTS dilations,
    #    @INT group, @INTS kernel_shape, @INTS output_shape, @INTS pads, @INTS strides)

    # Note: inputs are reversed from what one would expect.
    kernel_shape = conv_kernel_shape(ctx, node, 1)
    output_shape = node.inputs[0].get_tensor_value()
    if node.is_nhwc():
        new_output_shape = [output_shape[1], output_shape[2]]
    else:
        new_output_shape = [output_shape[2], output_shape[3]]
    node.set_attr("output_shape", new_output_shape)

    strides = conv_dims_attr(node, "strides")
    conv_dims_attr(node, "dilations")
    add_padding(node, kernel_shape, strides)

    # remove output_shapes input, swap data and kernel
    ctx.remove_input(node, node.input[0])
    t = node.input[0]
    node.input[0] = node.input[1]
    node.input[1] = t

    nodes = conv_convert_inputs(ctx, node, with_kernel=True)
    return nodes


def depthwiseconv_op(ctx, node, name, args):
    # T output = DepthwiseConv2dNative(T input, T filter, @list(int) strides, @string padding, @string data_format)
    # T Y = ConvTranspose(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
    #           @AttrType.INTS kernel_shape, @AttrType.INTS output_shape, @AttrType.INTS pads, @AttrType.INTS strides)
    #
    # this is clearly documented in onnx, the hint comes from pytorch documentation:
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
    add_padding(node, kernel_shape, strides)

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

    add_padding(node, kernel_shape, strides)

    nodes = conv_convert_inputs(ctx, node, with_kernel=False)
    return nodes


def relu6_op(ctx, node, name, args):
    # min(max(features, 0), 6)
    node.type = "Max"
    shape = ctx.get_shape(node.input[0])
    zero_name = utils.make_name(node.name)
    zero_node = ctx.make_const(zero_name, "Const", np.zeros(shape, dtype=np.float32))
    six_name = utils.make_name(node.name)
    six = np.zeros(shape, dtype=np.float32)
    six.fill(6.0)
    six_node = ctx.make_const(six_name, "Const", six)
    node.input.append(zero_name)
    op_name = utils.make_name(node.name)
    new_op = ctx.insert_new_node_on_output("Min", node.output[0], name=op_name)
    new_op.input.append(six_name)
    ctx.copy_shape(node.output[0], new_op.output[0])
    return [node, new_op]


def squareddifference_op(ctx, node, name, args):
    node.type = "Sub"
    op_name = utils.make_name(node.name)
    mul = ctx.insert_new_node_on_output("Mul", node.output[0], name=op_name)
    mul.input.append(node.output[0])
    return [node, mul]


def cast_op(ctx, node, name, args):
    # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
    # T2 output = Cast(T1 input, @STRING to)
    # already done in pass1
    return node


def biasadd_op(ctx, node, name, args):
    # T output = BiasAdd(T value, T bias, @string data_format)
    # T output = BiasAddV1(T value, T bias)
    # TODO: for now use add. We may need to convert to NCHW.
    node.type = "Add"
    return broadcast_op(ctx, node, name, args)


def transpose_op(ctx, node, name, args):
    # T y = Transpose(T x, Tperm perm, @type Tperm)
    # T transposed = Transpose(T data, @INTS perm)
    if len(node.input) > 1:
        shape = ctx.get_shape(node.inputs[1])
        ctx.remove_input(node, node.input[1])
        dims = [i for i in range(len(shape) - 1, 0)]
        node.set_attr("perm", dims)
    else:
        # perm comes as attribute from tensorflow
        pass
    return node


def concat_op(ctx, node, name, args):
    # T output = ConcatV2(T values, Tidx axis, @int N, @type Tidx)
    # T concat_result = Concat(T inputs, @INT axis)
    axis_node = node.inputs[-1]
    axis = axis_node.get_tensor_value()
    ctx.remove_input(node, node.input[-1])
    node.set_attr("axis", axis[0])
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


def gather_op(ctx, node, name, args):
    # Tparams output = Gather(Tparams params, Tindices indices, @bool validate_indices, @type Tparams, @type Tindices)
    # T output = Gather(T data, Tind indices, @INT axis)
    return node


def split_op(ctx, node, name, args):
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
    # T output = Pad(T input, Tpaddings paddings, @type Tpaddings)
    # T output = Pad(T data, @STRING mode, @INTS pads, @FLOAT value)
    paddings = np.array(node.inputs[1].get_tensor_value()).transpose().flatten()
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
    # tensorflow already inferes the output shape so we can just take it
    shape = ctx.get_shape(node.output[0])
    node.type = "Reshape"
    ctx.remove_input(node, node.input[1])
    node.set_attr("shape", shape)
    return node


def stridedslice_op(ctx, node, name, args):
    # T output = StridedSlice(T input, Index begin, Index end, Index strides,
    #               @type Index, @int begin_mask, @int end_mask, @int ellipsis_mask,
    #               @int new_axis_mask, @int shrink_axis_mask)
    # FIXME: needed by ops like tf.flatten()
    raise ValueError("stridedslice_op not implemented")
    return node


def pow_op(ctx, node, name, args):
    if "onnxmsrt" not in FLAVOR:
        return node
    # workaround a bug in onnxmsrt, pow(a, b) becomes np.exp(np.log(a) * b)
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


# pylint: enable=W0613,C0111,W0612

# map tensorflow ops to onnx ops. The format below is
# "TFOP": func_to_map, ["OnnxOp", ...]
#
_OPS_MAPPING = {
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
    "ConcatV2": (concat_op, ["Concat"]),
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
    "Gather": (gather_op, []),
    "Greater": (broadcast_op, []),
    "Identity": (identity_op, ["Identity"]),
    "Less": (broadcast_op, []),
    "Log": (direct_op, []),
    "LRN": (lrn_op, []),
    "LogicalAnd": (broadcast_op, ["And"]),
    "Max": (reduce_op, ["ReduceMax"]),
    "MatMul": (direct_op, ["MatMul"]),
    "Maximum": (direct_op, ["Max"]),
    "MaxPool": (pool_op, ["MaxPool"]),
    "MaxPoolV2": (pool_op, ["MaxPool"]),
    "Mean": (reduce_op, ["ReduceMean"]),
    "Min": (reduce_op, ["ReduceMin"]),
    "Minimum": (direct_op, ["Min"]),
    "Mul": (broadcast_op, []),
    "Neg": (direct_op, []),
    "NoOp": (no_op, []),
    "NotEqual": (direct_op, ["Not"]),
    "Pad": (pad_op, []),
    "Placeholder": (placeholder_op, []),
    "PlaceholderV2": (placeholder_op, []),
    "Pow": (pow_op, []),
    "Prod": (reduce_op, ["ReduceProd"]),
    "RandomNormal": (direct_op, []),
    "RandomUniform": (direct_op, []),
    "RealDiv": (broadcast_op, ["Div"]),
    "Reciprocal": (direct_op, []),
    "Relu": (direct_op, ["Relu"]),
    "Relu6": (relu6_op, []),
    "Reshape": (reshape_op, ["Reshape"]),
    "Rsqrt": (rsqrt_op, []),
    "Shape": (shape_op, []),
    "Sigmoid": (direct_op, []),
    "Slice": (slice_op, []),
    "SplitV": (split_op, ["Split"]),
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
}


def rewrite_random_uniform(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', inputs=[
                OpTypePattern('RandomUniform', name='input1'),
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
        shape = g.get_shape(output.output[0])
        dtype = output.dtype
        op_name = utils.make_name("RandomUniform")
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("RandomUniform", [], [out_name],
                                         name=op_name, low=tmin, high=tmax,
                                         dtype=dtype, shape=shape), g)
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
        shape = g.get_shape(output.output[0])
        dtype = output.dtype
        op_name = utils.make_name("RandomNormal")
        out_name = op_name + ":0"
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
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("Dropout", [inputs2.input[0]], [out_name], name=op_name, ratio=1.0), g)
        ops = g.replace_subgraph(ops, match, [inputs2], [outputs], [new_node], [new_node])

    return ops


def rewrite_flatten(g, ops):
    pattern = \
        OpTypePattern('Reshape', name='outputs', inputs=[
            OpTypePattern("*", name="input2"),
            OpTypePattern('Pack', inputs=[
                OpTypePattern('StridedSlice', inputs=[
                    OpTypePattern('Shape', name="input1", inputs=["*"]),
                    "*", "*", "*"
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
        out_name = op_name + ":0"
        new_node = Node(helper.make_node("Flatten", [inputs2.output[0]], [out_name], name=op_name, ratio=1.0), g)
        g.replace_all_inputs(ops, outputs.output[0], out_name)
        to_be_removed = [node for node in match.get_nodes() if node != inputs2]
        for i in range(len(ops)-1, -1, -1):
            if ops[i] in to_be_removed:
                del ops[i]
        ops.append(new_node)
    return ops


def tensorflow_onnx_mapping(g, continue_on_error):
    mapped_op = collections.Counter()
    unmapped_op = collections.Counter()

    ops = g.get_nodes()
    onnx_nodes = []
    # noinspection PyNoneFunctionAssignment,PyNoneFunctionAssignment
    for node in ops:
        op = node.type
        map_info = _OPS_MAPPING.get(op)
        if map_info is None:
            if continue_on_error:
                unmapped_op[op] += 1
                continue
            else:
                raise ValueError("tensorflow op " + op + " is not supported")
        else:
            mapped_op[op] += 1
            func, args = map_info

        if args:
            node.type = args[0]
            args = args[1:]
        try:
            onnx_node = func(g, node, node.name, args)
        except Exception as ex:
            raise ex
        if onnx_node:
            if isinstance(onnx_node, list):
                onnx_nodes.extend(onnx_node)
            else:
                onnx_nodes.append(onnx_node)

    g.set_nodes(onnx_nodes)
    return mapped_op, unmapped_op


def tf_optimize(sess, inputs, outputs, graph_def):
    """Optimize tensorflow graph for inference."""
    transforms = [
        # "remove_nodes(op=Identity, op=CheckNumerics)",
        "fold_batch_norms",
        "fold_old_batch_norms"
        # fails: "fold_constants(ignore_errors=true)",
    ]
    needed_names = [utils.node_name(i) for i in inputs] + [utils.node_name(i) for i in outputs]
    graph_def = graph_util.extract_sub_graph(graph_def, needed_names)
    graph_def = TransformGraph(graph_def, inputs, outputs, transforms)
    return graph_def


def tf_freeze(input_graph, input_checkpoint, output_graph, output_node_names):
    """Freeze tensorflow graph."""
    freeze_graph(input_graph=input_graph,
                 input_saver="",
                 input_binary=False,
                 input_checkpoint=input_checkpoint,
                 output_node_names=output_node_names,
                 restore_op_name="save/restore_all",
                 filename_tensor_name="save/Const:0",
                 output_graph=output_graph,
                 clear_devices=True,
                 initializer_nodes="")


def optimize_onnxgraph(ctx, ops):
    """
    Optimize onnx graph.
    For now remove equalizing transposes
    """
    # TODO: remove Transose->Relu->Transpose
    ret_ops = []
    for node in ops:
        if node.type == "Transpose" and node.input and node.inputs[0] and node.inputs[0].type == "Transpose":
            prev = node.inputs[0]
            p1 = node.get_attr("perm").ints
            p2 = prev.get_attr("perm").ints
            if p1 == NHWC_TO_NCHW and p2 == NCHW_TO_NHWC:
                input_name = prev.input[0]
                output_name = node.output[0]
                for next_node in ops:
                    for i, n in enumerate(next_node.input):
                        if n == output_name:
                            next_node.input[i] = input_name
                del prev
        else:
            ret_ops.append(node)
    ctx.set_nodes(ret_ops)


def process_tf_graph(graph, continue_on_error=False, verbose=False):
    """Convert tensorflow graph to onnx graph."""

    onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes = \
        tensorflow_to_onnx(graph)

    g = Graph(onnx_nodes, output_shapes, dtypes)
    ops = g.get_nodes()

    # rewrites
    for rewrite in [rewrite_flatten,
                    rewrite_random_uniform,
                    rewrite_random_normal,
                    rewrite_dropout,
                    rewrite_transpose]:
        ops = rewrite(g, ops)
        g.set_nodes(ops)

    g.topological_sort(g.get_nodes())
    mapped_op, unmapped_op = tensorflow_onnx_mapping(g, continue_on_error)
    g.topological_sort(g.get_nodes())
    optimize_onnxgraph(g, g.get_nodes())

    g.update_proto()
    if verbose:
        print("tensorflow ops: {}".format(op_cnt))
        print("tensorflow attr: {}".format(attr_cnt))
        print("onnx mapped: {}".format(mapped_op))
        print("onnx unmapped: {}".format(unmapped_op))
    return g
