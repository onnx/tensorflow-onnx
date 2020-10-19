
from tf2onnx.handler import tfl_op
from tf2onnx import constants, utils
import numpy as np

@tfl_op(["TFL_TRANSPOSE_CONV"], tf_op="Conv2DBackpropInput")
class TflTransposeConv:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        # No need to change 'padding' attribute
        stride_h = node.get_attr_int("stride_h") # TODO: Permute this?
        stride_w = node.get_attr_int("stride_w")
        node.set_attr("strides", [1, stride_h, stride_w, 1])
        del node.attr["stride_h"]
        del node.attr["stride_w"]
        transpose_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[1], name=None, perm=[1, 2, 0, 3])
        transpose_node.skip_conversion = True
        node.set_attr("data_format", "NHWC")

@tfl_op(["TFL_CONV_2D"], tf_op="Conv2D")
class TflConv2D:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")
        # No need to change 'padding' attribute
        stride_h = node.get_attr_int("stride_h") # TODO: Permute this?
        stride_w = node.get_attr_int("stride_w")
        dilation_w_factor = node.get_attr_int("dilation_w_factor")
        dilation_h_factor = node.get_attr_int("dilation_h_factor")
        node.set_attr("strides", [1, stride_h, stride_w, 1])
        node.set_attr("dilations", [1, dilation_h_factor, dilation_w_factor, 1])
        del node.attr["stride_h"]
        del node.attr["stride_w"]
        del node.attr["dilation_h_factor"]
        del node.attr["dilation_w_factor"]
        transpose_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[1], name=None, perm=[1, 2, 3, 0])
        transpose_node.skip_conversion = True
        node.set_attr("data_format", "NHWC")

@tfl_op(["TFL_AVERAGE_POOL_2D"], tf_op="AvgPool")
@tfl_op(["TFL_MAX_POOL_2D"], tf_op="MaxPool")
class TflAveragePool:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")
        # No need to change 'padding' attribute
        stride_h = node.get_attr_int("stride_h") # TODO: Permute this?
        stride_w = node.get_attr_int("stride_w")
        filter_height = node.get_attr_int("filter_height") # TODO: Permute this?
        filter_width = node.get_attr_int("filter_width")
        node.set_attr("strides", [1, stride_h, stride_w, 1])
        node.set_attr("ksize", [1, filter_height, filter_width, 1])
        del node.attr["stride_h"]
        del node.attr["stride_w"]
        del node.attr["filter_height"]
        del node.attr["filter_width"]
        node.set_attr("data_format", "NHWC")

@tfl_op(["TFL_DEPTHWISE_CONV_2D"], tf_op="DepthwiseConv2dNative")
class TflDepthwiseConv2D:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        fused_activation_function = node.attr['fused_activation_function'].s
        depth_multiplier = node.get_attr_int('depth_multiplier') # TODO: use this?
        utils.make_sure(fused_activation_function == b'NONE', "fused_activation_function not yet supported")
        # No need to change 'padding' attribute
        stride_h = node.get_attr_int("stride_h") # TODO: Permute this?
        stride_w = node.get_attr_int("stride_w")
        dilation_w_factor = node.get_attr_int("dilation_w_factor")
        dilation_h_factor = node.get_attr_int("dilation_h_factor")
        node.set_attr("strides", [1, stride_h, stride_w, 1])
        node.set_attr("dilations", [1, dilation_h_factor, dilation_w_factor, 1])
        del node.attr["stride_h"]
        del node.attr["stride_w"]
        del node.attr["dilation_h_factor"]
        del node.attr["dilation_w_factor"]
        transpose_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[1], name=None, perm=[1, 2, 3, 0])
        transpose_node.skip_conversion = True
        node.set_attr("data_format", "NHWC")

@tfl_op(["TFL_BATCH_TO_SPACE_ND"], tf_op="BatchToSpaceND")
class TflSlice:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass